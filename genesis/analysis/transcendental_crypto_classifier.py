"""
Clasificador trascendental de criptomonedas con estrategia adaptativa.

Este módulo implementa un sistema avanzado de clasificación de criptomonedas
que mantiene su eficiencia independientemente del tamaño del capital,
aplicando estrategias adaptativas para compensar los efectos de escala.

Características principales:
- Análisis multidimensional con 7 factores clave
- Compensación dinámica por efectos del tamaño del capital
- Distribución inteligente entre exchanges para mantener eficiencia
- Detección de umbrales de saturación de liquidez
- Proyección de desempeño en diferentes niveles de capital
"""
import asyncio
import logging
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union, cast

import numpy as np
import pandas as pd
from sqlalchemy import select, insert, update, delete, and_, or_, desc

from genesis.db.models.crypto_classifier_models import (
    Cryptocurrency, CryptoClassification, CryptoMetrics,
    ClassificationHistory, CapitalScaleEffect
)
from genesis.db.transcendental_database import transcendental_db
from genesis.db.base import db_manager

# Configuración de logging
logger = logging.getLogger("genesis.analysis.crypto_classifier")

# Constantes globales
DEFAULT_CAPITAL_BASE = 10000.0  # $10,000 USD
DEFAULT_CONFIDENCE_THRESHOLD = 0.7  # Umbral de confianza para clasificaciones
DEFAULT_EXCHANGES = ["binance", "kucoin", "bybit", "okx", "coinbase"]
MIN_DATA_POINTS = 30  # Mínimo de puntos de datos necesarios para clasificación confiable


class AdaptiveScoreAdjuster:
    """
    Ajustador de puntuaciones para compensar efectos del tamaño del capital.
    
    Este componente aplica ajustes a las puntuaciones de clasificación basados
    en el tamaño del capital, para mantener la precisión y eficiencia del sistema
    independientemente de la escala.
    """
    
    def __init__(self, base_capital: float = DEFAULT_CAPITAL_BASE):
        """
        Inicializar ajustador adaptativo.
        
        Args:
            base_capital: Capital base para los cálculos
        """
        self.base_capital = base_capital
        self.scale_factor_cache: Dict[str, Dict[str, float]] = {}
        self.saturation_points: Dict[str, float] = {}
    
    def adjust_score(
        self, 
        symbol: str, 
        raw_scores: Dict[str, float], 
        capital: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Ajustar puntuaciones en función del capital.
        
        Args:
            symbol: Símbolo de la criptomoneda
            raw_scores: Puntuaciones originales por factor
            capital: Capital actual
            metrics: Métricas adicionales
            
        Returns:
            Puntuaciones ajustadas
        """
        # Si el capital es igual o menor al base, no hay ajuste
        if capital <= self.base_capital:
            return raw_scores.copy()
        
        adjusted_scores = {}
        scale_ratio = capital / self.base_capital
        
        # Obtener o calcular factores de escala para el símbolo
        scale_factors = self._get_scale_factors(symbol, metrics)
        
        # Aplicar ajustes específicos por factor
        for factor, score in raw_scores.items():
            if factor in scale_factors:
                # Fórmula: score_ajustado = score_original * (1 - factor_escala * log(escala))
                scale_impact = scale_factors[factor] * np.log10(scale_ratio)
                
                # Limitar el impacto máximo al 95%
                scale_impact = min(scale_impact, 0.95)
                
                adjusted_scores[factor] = score * (1 - scale_impact)
            else:
                adjusted_scores[factor] = score
        
        return adjusted_scores
    
    def _get_scale_factors(self, symbol: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Obtener factores de escala para una criptomoneda.
        
        Args:
            symbol: Símbolo de la criptomoneda
            metrics: Métricas disponibles
            
        Returns:
            Factores de escala por componente
        """
        # Usar cache si existe
        if symbol in self.scale_factor_cache:
            return self.scale_factor_cache[symbol]
        
        # Calcular factores de escala basados en métricas
        scale_factors = {
            "liquidity_score": 0.15,  # Factor base para liquidez
            "alpha_score": 0.05,      # El alfa se ve menos afectado
            "volatility_score": 0.02,  # La volatilidad casi no cambia con escala
            "momentum_score": 0.10,   # El momentum se ve moderadamente afectado
            "trend_score": 0.03,      # La tendencia se ve poco afectada
            "correlation_score": 0.01,  # La correlación casi no cambia
            "exchange_quality_score": 0.08  # Calidad del exchange
        }
        
        # Ajustar en función de métricas si están disponibles
        if metrics:
            # La liquidez es el factor más sensible a la escala
            if metrics.get("orderbook_depth_usd"):
                # A menor profundidad, mayor sensibilidad a escala
                depth_factor = min(1.0, 1000000 / max(10000, float(metrics["orderbook_depth_usd"])))
                scale_factors["liquidity_score"] = 0.15 * (1 + depth_factor)
            
            # El deslizamiento indica problemas de escala
            if metrics.get("slippage_10000usd"):
                slippage = float(metrics["slippage_10000usd"])
                if slippage > 0.001:  # 0.1% de deslizamiento
                    slippage_factor = min(3.0, slippage * 1000)
                    scale_factors["liquidity_score"] *= slippage_factor
        
        # Guardar en cache
        self.scale_factor_cache[symbol] = scale_factors
        return scale_factors
    
    def estimate_saturation_point(
        self, 
        symbol: str, 
        metrics: Dict[str, Any]
    ) -> float:
        """
        Estimar punto de saturación de capital para una criptomoneda.
        
        Este es el punto donde añadir más capital no mejora los rendimientos
        debido a limitaciones de liquidez.
        
        Args:
            symbol: Símbolo de la criptomoneda
            metrics: Métricas disponibles
            
        Returns:
            Punto de saturación estimado en USD
        """
        if symbol in self.saturation_points:
            return self.saturation_points[symbol]
        
        # Valor base de saturación
        base_saturation = 1_000_000  # $1M por defecto
        
        # Ajustar según métricas
        if metrics:
            # Profundidad del libro de órdenes
            if metrics.get("orderbook_depth_usd"):
                depth = float(metrics["orderbook_depth_usd"])
                # Punto de saturación aproximado = 5-10% de la profundidad total
                depth_saturation = depth * 0.08
                base_saturation = max(base_saturation, depth_saturation)
            
            # Volume 24h indica liquidez
            if metrics.get("volume_24h"):
                volume = float(metrics["volume_24h"])
                # Aprox. 1% del volumen diario es sostenible
                volume_saturation = volume * 0.01
                base_saturation = max(base_saturation, volume_saturation)
            
            # Market cap como indicador de tamaño
            if metrics.get("market_cap"):
                mcap = float(metrics["market_cap"])
                # 0.05% - 0.1% del market cap
                mcap_saturation = mcap * 0.0005
                base_saturation = min(base_saturation, mcap_saturation)
        
        # Limitar a rangos razonables
        saturation = max(100_000, min(1_000_000_000, base_saturation))
        
        # Guardar en cache
        self.saturation_points[symbol] = saturation
        return saturation
    
    def get_optimal_capital_per_symbol(
        self, 
        symbols: List[str],
        metrics_dict: Dict[str, Dict[str, Any]],
        total_capital: float
    ) -> Dict[str, float]:
        """
        Distribuir el capital de manera óptima entre criptomonedas.
        
        Args:
            symbols: Lista de símbolos
            metrics_dict: Métricas por símbolo
            total_capital: Capital total a distribuir
            
        Returns:
            Asignación óptima de capital por símbolo
        """
        # Calcular saturación para cada símbolo
        saturations = {
            s: self.estimate_saturation_point(s, metrics_dict.get(s, {}))
            for s in symbols
        }
        
        # Calcular asignación inicial basada en saturación
        total_saturation = sum(saturations.values())
        allocations = {}
        
        for symbol in symbols:
            sat_weight = saturations[symbol] / total_saturation if total_saturation > 0 else 1.0 / len(symbols)
            allocations[symbol] = total_capital * sat_weight
            
            # Limitar a la saturación
            if allocations[symbol] > saturations[symbol]:
                allocations[symbol] = saturations[symbol]
        
        # Ajustar para alcanzar el capital total
        allocated = sum(allocations.values())
        if allocated < total_capital:
            remaining = total_capital - allocated
            
            # Distribuir el restante a los no saturados
            non_saturated = [s for s in symbols if allocations[s] < saturations[s]]
            
            if non_saturated:
                for symbol in non_saturated:
                    space_left = saturations[symbol] - allocations[symbol]
                    share = space_left / sum(saturations[s] - allocations[s] for s in non_saturated)
                    allocations[symbol] += remaining * share
        
        return allocations
    
    def get_optimal_exchange_distribution(
        self, 
        symbol: str, 
        capital: float,
        exchanges: List[str],
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Distribuir capital entre exchanges de manera óptima.
        
        Args:
            symbol: Símbolo de la criptomoneda
            capital: Capital a distribuir
            exchanges: Exchanges disponibles
            metrics: Métricas por exchange
            
        Returns:
            Distribución óptima por exchange
        """
        # Distribución base equitativa
        distribution = {ex: 1.0 for ex in exchanges}
        
        # Ajustar según métricas si están disponibles
        if metrics and metrics.get("exchange_metrics"):
            ex_metrics = metrics["exchange_metrics"]
            
            # Normalizar métricas
            total_weight = 0
            for ex in exchanges:
                if ex in ex_metrics:
                    # Combinar liquidez, volumen y calidad
                    liquidity = ex_metrics[ex].get("liquidity", 0.5)
                    volume = ex_metrics[ex].get("volume", 0.5)
                    quality = ex_metrics[ex].get("quality", 0.5)
                    
                    # Peso compuesto
                    weight = (liquidity * 0.5) + (volume * 0.3) + (quality * 0.2)
                    distribution[ex] = weight
                    total_weight += weight
            
            # Normalizar pesos
            if total_weight > 0:
                for ex in distribution:
                    distribution[ex] = distribution[ex] / total_weight
        else:
            # Sin métricas, distribución equitativa
            even_weight = 1.0 / len(exchanges)
            for ex in exchanges:
                distribution[ex] = even_weight
        
        # Multiplicar por el capital para obtener asignación
        allocation = {ex: capital * weight for ex, weight in distribution.items()}
        
        return allocation


class TranscendentalCryptoClassifier:
    """
    Clasificador transcendental de criptomonedas con estrategia adaptativa.
    
    Este clasificador implementa la estrategia adaptativa que mantiene
    su eficacia independientemente del tamaño del capital, mediante
    ajustes dinámicos basados en factores multidimensionales.
    """
    
    def __init__(
        self, 
        initial_capital: float = DEFAULT_CAPITAL_BASE,
        exchanges: Optional[List[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        """
        Inicializar clasificador transcendental.
        
        Args:
            initial_capital: Capital inicial para clasificación base
            exchanges: Lista de exchanges a considerar
            confidence_threshold: Umbral de confianza para clasificaciones
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.confidence_threshold = confidence_threshold
        self.exchanges = exchanges or DEFAULT_EXCHANGES
        
        self.score_adjuster = AdaptiveScoreAdjuster(base_capital=initial_capital)
        self.last_classification_time = datetime.now() - timedelta(days=1)
        self.classified_symbols: Set[str] = set()
        self.hot_cryptos: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"TranscendentalCryptoClassifier inicializado con capital={initial_capital}, "
            f"{len(self.exchanges)} exchanges"
        )
    
    async def classify_all(self, capital: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Clasificar todas las criptomonedas disponibles.
        
        Args:
            capital: Capital actual (usa el inicial si no se especifica)
            
        Returns:
            Lista de clasificaciones
        """
        # Actualizar capital si se especifica
        if capital is not None and capital > 0:
            self.current_capital = capital
        
        logger.info(f"Iniciando clasificación con capital={self.current_capital}")
        
        # Obtener todas las criptomonedas activas
        def get_active_cryptos_query():
            return (
                "SELECT * FROM cryptocurrencies WHERE is_active = :is_active ORDER BY market_cap DESC",
                {"is_active": True}
            )
        
        cryptos = await transcendental_db.execute_query(get_active_cryptos_query)
        
        results = []
        start_time = time.time()
        
        # Procesar por lotes para mejor rendimiento
        batch_size = 10
        for i in range(0, len(cryptos), batch_size):
            batch = cryptos[i:i+batch_size]
            tasks = [self.classify_crypto(crypto) for crypto in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend([r for r in batch_results if r])
        
        # Actualizar hora de última clasificación
        self.last_classification_time = datetime.now()
        
        # Actualizar conjunto de símbolos clasificados
        self.classified_symbols = {r["symbol"] for r in results}
        
        # Actualizar hot_cryptos
        self.hot_cryptos = {
            r["symbol"]: r for r in results if r.get("is_hot", False)
        }
        
        logger.info(
            f"Clasificación completada en {time.time() - start_time:.2f}s: "
            f"{len(results)} criptomonedas, {len(self.hot_cryptos)} hot"
        )
        
        return results
    
    async def classify_crypto(self, crypto: Any) -> Optional[Dict[str, Any]]:
        """
        Clasificar una única criptomoneda.
        
        Args:
            crypto: Objeto Cryptocurrency
            
        Returns:
            Clasificación como diccionario o None si no hay suficientes datos
        """
        symbol = crypto.symbol
        
        # Obtener métricas actuales
        metrics = await self._get_crypto_metrics(crypto.id)
        
        if not metrics:
            logger.warning(f"No hay métricas disponibles para {symbol}, omitiendo clasificación")
            return None
        
        # Calcular puntuaciones por factor
        raw_scores = self._calculate_factor_scores(crypto, metrics)
        
        # Aplicar ajustes adaptativos basados en el capital actual
        adjusted_scores = self.score_adjuster.adjust_score(
            symbol, raw_scores, self.current_capital, metrics
        )
        
        # Calcular puntuación final combinada
        weights = {
            "alpha_score": 0.20,
            "liquidity_score": 0.25,
            "volatility_score": 0.15,
            "momentum_score": 0.10,
            "trend_score": 0.15,
            "correlation_score": 0.10,
            "exchange_quality_score": 0.05
        }
        
        final_score = sum(adjusted_scores[factor] * weights[factor] for factor in adjusted_scores)
        
        # Determinar si es "hot" (usando umbral dinámico)
        hot_threshold = 0.75  # Umbral base
        
        # Ajustar umbral según tamaño de capital
        if self.current_capital > self.initial_capital:
            capital_ratio = min(10, self.current_capital / self.initial_capital)
            # Mayor capital = umbral más exigente
            hot_threshold += 0.05 * np.log10(capital_ratio)
        
        is_hot = final_score >= hot_threshold
        
        # Crear clasificación en base de datos
        classification_id = await self._store_classification(
            crypto.id, raw_scores, adjusted_scores, final_score, is_hot
        )
        
        # Estimar punto de saturación
        saturation_point = self.score_adjuster.estimate_saturation_point(symbol, metrics)
        
        # Calcular distribución óptima entre exchanges
        exchange_distribution = self.score_adjuster.get_optimal_exchange_distribution(
            symbol, min(self.current_capital * 0.1, saturation_point * 0.5), self.exchanges, metrics
        )
        
        # Compilar resultado
        result = {
            "symbol": symbol,
            "name": crypto.name,
            "classification_id": classification_id,
            "raw_scores": raw_scores,
            "adjusted_scores": adjusted_scores,
            "final_score": final_score,
            "is_hot": is_hot,
            "saturation_point": saturation_point,
            "capital_efficiency": 1.0 if self.current_capital <= saturation_point else saturation_point / self.current_capital,
            "exchange_distribution": exchange_distribution,
            "market_cap": crypto.market_cap,
            "current_price": crypto.current_price,
            "classification_time": datetime.now().isoformat()
        }
        
        return result
    
    async def get_hot_cryptos(self, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Obtener criptomonedas clasificadas como "hot".
        
        Args:
            refresh: Si se debe refrescar la clasificación
            
        Returns:
            Diccionario con criptomonedas hot
        """
        # Si se solicita refresh o ha pasado más de 1 hora desde la última clasificación
        if (
            refresh or 
            not self.hot_cryptos or 
            (datetime.now() - self.last_classification_time) > timedelta(hours=1)
        ):
            await self.classify_all()
        
        return self.hot_cryptos
    
    async def analyze_capital_scaling(
        self, 
        symbol: str, 
        capital_levels: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analizar cómo se comporta la clasificación con diferentes niveles de capital.
        
        Args:
            symbol: Símbolo de la criptomoneda
            capital_levels: Niveles de capital a evaluar
            
        Returns:
            Resultados del análisis de escalabilidad
        """
        # Niveles de capital por defecto
        if not capital_levels:
            capital_levels = [10_000, 100_000, 1_000_000, 10_000_000]
        
        # Obtener criptomoneda y métricas
        async def get_crypto_query():
            return (
                "SELECT * FROM cryptocurrencies WHERE symbol = :symbol",
                {"symbol": symbol}
            )
        
        cryptos = await transcendental_db.execute_query(get_crypto_query)
        if not cryptos:
            return {"error": f"Criptomoneda {symbol} no encontrada"}
        
        crypto = cryptos[0]
        metrics = await self._get_crypto_metrics(crypto.id)
        
        if not metrics:
            return {"error": f"No hay métricas disponibles para {symbol}"}
        
        # Calcular puntuaciones base
        raw_scores = self._calculate_factor_scores(crypto, metrics)
        
        # Analizar para cada nivel de capital
        scores_by_capital = {}
        for capital in capital_levels:
            adjusted_scores = self.score_adjuster.adjust_score(
                symbol, raw_scores, capital, metrics
            )
            
            # Calcular puntuación final con pesos
            weights = {
                "alpha_score": 0.20,
                "liquidity_score": 0.25,
                "volatility_score": 0.15,
                "momentum_score": 0.10,
                "trend_score": 0.15,
                "correlation_score": 0.10,
                "exchange_quality_score": 0.05
            }
            
            final_score = sum(adjusted_scores[factor] * weights[factor] for factor in adjusted_scores)
            scores_by_capital[str(capital)] = final_score
        
        # Calcular sensibilidad a escala
        base_score = scores_by_capital[str(capital_levels[0])]
        max_score = base_score
        min_score = base_score
        
        for score in scores_by_capital.values():
            max_score = max(max_score, score)
            min_score = min(min_score, score)
        
        scale_sensitivity = 0.0
        if base_score > 0:
            scale_sensitivity = (base_score - min_score) / base_score
        
        # Estimar punto de saturación
        saturation_point = self.score_adjuster.estimate_saturation_point(symbol, metrics)
        
        # Determinar factor limitante
        factors = list(raw_scores.keys())
        factor_sensitivities = {}
        
        for factor in factors:
            temp_scores = raw_scores.copy()
            temp_scores[factor] = temp_scores[factor] * 2  # Duplicar este factor
            
            # Ver cómo afecta a los diferentes niveles
            impact = 0
            for capital in capital_levels:
                adj_normal = self.score_adjuster.adjust_score(
                    symbol, raw_scores, capital, metrics
                )
                adj_modified = self.score_adjuster.adjust_score(
                    symbol, temp_scores, capital, metrics
                )
                
                # Diferencia relativa
                normal_weighted = sum(adj_normal[f] * weights[f] for f in adj_normal)
                modified_weighted = sum(adj_modified[f] * weights[f] for f in adj_modified)
                
                impact += abs(modified_weighted - normal_weighted) / max(normal_weighted, 0.001)
            
            factor_sensitivities[factor] = impact / len(capital_levels)
        
        # El factor con mayor sensibilidad es el limitante
        limiting_factor = max(factor_sensitivities.items(), key=lambda x: x[1])[0]
        
        # Almacenar análisis en base de datos
        await self._store_capital_scale_analysis(
            crypto.id, 
            {str(c): s for c, s in zip(capital_levels, [scores_by_capital[str(c)] for c in capital_levels])},
            scale_sensitivity,
            limiting_factor,
            saturation_point
        )
        
        return {
            "symbol": symbol,
            "raw_scores": raw_scores,
            "scores_by_capital": scores_by_capital,
            "scale_sensitivity": scale_sensitivity,
            "saturation_point": saturation_point,
            "limiting_factor": limiting_factor,
            "factor_sensitivities": factor_sensitivities,
            "analysis_time": datetime.now().isoformat()
        }
    
    async def update_capital(self, new_capital: float) -> Dict[str, Any]:
        """
        Actualizar el capital usado para clasificaciones.
        
        Args:
            new_capital: Nuevo capital en USD
            
        Returns:
            Resultados de la actualización
        """
        if new_capital <= 0:
            return {"error": "El capital debe ser positivo"}
        
        old_capital = self.current_capital
        self.current_capital = new_capital
        
        logger.info(f"Capital actualizado: {old_capital} -> {new_capital}")
        
        # Si el capital cambia significativamente, refrescar clasificaciones
        if abs(new_capital / old_capital - 1) > 0.1:  # Cambio > 10%
            logger.info(f"Cambio significativo de capital, refrescando clasificaciones")
            await self.classify_all()
            
            return {
                "previous_capital": old_capital,
                "new_capital": new_capital,
                "classifications_updated": True,
                "hot_cryptos_count": len(self.hot_cryptos)
            }
        
        return {
            "previous_capital": old_capital,
            "new_capital": new_capital,
            "classifications_updated": False
        }
    
    async def _get_crypto_metrics(self, crypto_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtener métricas para una criptomoneda.
        
        Args:
            crypto_id: ID de la criptomoneda
            
        Returns:
            Métricas como diccionario o None si no hay datos
        """
        # Definir consulta directamente como una tupla
        metrics_query = (
            "SELECT * FROM crypto_metrics WHERE cryptocurrency_id = :crypto_id ORDER BY updated_at DESC LIMIT 1",
            {"crypto_id": crypto_id}
        )
        
        metrics_result = await transcendental_db.execute_query(metrics_query)
        
        if not metrics_result:
            return None
        
        metrics_obj = metrics_result[0]
        return metrics_obj.to_dict()
    
    def _calculate_factor_scores(self, crypto: Any, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcular puntuaciones por factor para una criptomoneda.
        
        Args:
            crypto: Objeto Cryptocurrency
            metrics: Métricas disponibles
            
        Returns:
            Puntuaciones por factor
        """
        scores = {}
        
        # Alpha score (rentabilidad ajustada por riesgo)
        alpha = 0.5  # Valor por defecto
        if metrics.get("sharpe_ratio"):
            sharpe = float(metrics["sharpe_ratio"])
            # Normalizar: 2+ es excelente, 1 es bueno, 0 es neutral, <0 es malo
            alpha = min(1.0, max(0.0, (sharpe + 0.5) / 2.5))
        scores["alpha_score"] = alpha
        
        # Liquidity score
        liquidity = 0.5  # Valor por defecto
        if metrics.get("orderbook_depth_usd") and metrics.get("slippage_10000usd"):
            depth = float(metrics["orderbook_depth_usd"])
            slippage = float(metrics["slippage_10000usd"])
            
            # Normalizar: profundidad > $5M es excelente
            depth_score = min(1.0, depth / 5_000_000)
            
            # Normalizar: slippage < 0.1% es excelente
            slippage_score = min(1.0, max(0.0, 1.0 - slippage * 10))
            
            # Combinar (profundidad 60%, slippage 40%)
            liquidity = (depth_score * 0.6) + (slippage_score * 0.4)
        scores["liquidity_score"] = liquidity
        
        # Volatility score (menor volatilidad = mejor puntuación)
        volatility = 0.5  # Valor por defecto
        if metrics.get("volatility_30d"):
            vol = float(metrics["volatility_30d"])
            # Normalizar: volatilidad < 50% anualizada es excelente
            # (aunque para estables, puede ser mucho menor)
            volatility = min(1.0, max(0.0, 1.0 - vol / 1.5))
        scores["volatility_score"] = volatility
        
        # Momentum score
        momentum = 0.5  # Valor por defecto
        if crypto.price_change_percentage_24h is not None:
            change_24h = float(crypto.price_change_percentage_24h)
            # Normalizar: >5% es excelente, <-5% es pésimo
            momentum = min(1.0, max(0.0, (change_24h + 5) / 10))
        scores["momentum_score"] = momentum
        
        # Trend score (fuerza de la tendencia)
        trend = 0.5  # Valor por defecto
        # Idealmente se calcularía con indicadores técnicos
        # Por ahora usamos un proxy basado en price_change + volatilidad
        if crypto.price_change_percentage_24h is not None and metrics.get("volatility_30d"):
            change_24h = float(crypto.price_change_percentage_24h)
            vol = float(metrics["volatility_30d"])
            
            # Tendencia = cambio normalizado por volatilidad
            raw_trend = abs(change_24h) / (vol * 100) if vol > 0 else 0
            trend = min(1.0, raw_trend * 5)  # Normalizar
        scores["trend_score"] = trend
        
        # Correlation score (menor correlación = mejor diversificación)
        correlation = 0.5  # Valor por defecto (neutral)
        # En un sistema real se calcularía la correlación con BTC/ETH
        # Por simplicidad, usamos un valor aleatorio estable por símbolo
        random.seed(crypto.symbol)
        correlation = min(1.0, max(0.0, 1.0 - (random.random() * 0.5 + 0.3)))
        scores["correlation_score"] = correlation
        
        # Exchange quality score
        exchange_quality = 0.5  # Valor por defecto
        if metrics.get("exchange_metrics"):
            ex_metrics = metrics["exchange_metrics"]
            quality_scores = [
                ex_metrics[ex].get("quality", 0.5) 
                for ex in ex_metrics 
                if ex in self.exchanges
            ]
            if quality_scores:
                exchange_quality = sum(quality_scores) / len(quality_scores)
        scores["exchange_quality_score"] = exchange_quality
        
        return scores
    
    async def _store_classification(
        self,
        crypto_id: int,
        raw_scores: Dict[str, float],
        adjusted_scores: Dict[str, float],
        final_score: float,
        is_hot: bool
    ) -> int:
        """
        Almacenar clasificación en la base de datos.
        
        Args:
            crypto_id: ID de la criptomoneda
            raw_scores: Puntuaciones originales
            adjusted_scores: Puntuaciones ajustadas
            final_score: Puntuación final
            is_hot: Si es una criptomoneda "hot"
            
        Returns:
            ID de la clasificación
        """
        # Verificar si ya existe una clasificación reciente (< 1 día)
        check_recent_query = (
            select(CryptoClassification)
            .where(
                and_(
                    CryptoClassification.cryptocurrency_id == crypto_id,
                    CryptoClassification.classification_date > datetime.now() - timedelta(days=1)
                )
            )
            .order_by(CryptoClassification.classification_date.desc())
            .limit(1)
        ), []
        
        recent = await transcendental_db.execute_query(check_recent_query)
        
        if recent:
            # Existe clasificación reciente, actualizarla
            classification = recent[0]
            old_score = classification.final_score
            old_hot = classification.hot_rating
            classification_id = classification.id
            
            # Calcular cambio de puntuación
            score_change_pct = ((final_score - old_score) / old_score * 100) if old_score > 0 else 0
            
            # Preparar actualización
            update_query = (
                update(CryptoClassification)
                .where(CryptoClassification.id == classification_id)
                .values(
                    alpha_score=adjusted_scores["alpha_score"],
                    liquidity_score=adjusted_scores["liquidity_score"],
                    volatility_score=adjusted_scores["volatility_score"],
                    momentum_score=adjusted_scores["momentum_score"],
                    trend_score=adjusted_scores["trend_score"],
                    correlation_score=adjusted_scores["correlation_score"],
                    exchange_quality_score=adjusted_scores["exchange_quality_score"],
                    final_score=final_score,
                    hot_rating=is_hot,
                    capital_base=self.current_capital,
                    classification_date=datetime.now(),
                    confidence=self.confidence_threshold
                )
            ), []
            
            await transcendental_db.execute_query(update_query)
            
            # Registrar cambio si es significativo o cambia estado hot
            if abs(score_change_pct) > 5 or old_hot != is_hot:
                await self._store_classification_history(
                    classification_id, old_score, final_score, old_hot, is_hot, score_change_pct
                )
        else:
            # No existe, crear nueva clasificación
            insert_query = (
                insert(CryptoClassification).values(
                    cryptocurrency_id=crypto_id,
                    alpha_score=adjusted_scores["alpha_score"],
                    liquidity_score=adjusted_scores["liquidity_score"],
                    volatility_score=adjusted_scores["volatility_score"],
                    momentum_score=adjusted_scores["momentum_score"],
                    trend_score=adjusted_scores["trend_score"],
                    correlation_score=adjusted_scores["correlation_score"],
                    exchange_quality_score=adjusted_scores["exchange_quality_score"],
                    final_score=final_score,
                    hot_rating=is_hot,
                    capital_base=self.current_capital,
                    classification_date=datetime.now(),
                    confidence=self.confidence_threshold
                ).returning(CryptoClassification.id)
            ), []
            
            result = await transcendental_db.execute_query(insert_query)
            classification_id = result[0][0]
        
        return classification_id
    
    async def _store_classification_history(
        self,
        classification_id: int,
        old_score: float,
        new_score: float,
        old_hot: bool,
        new_hot: bool,
        change_magnitude: float
    ) -> None:
        """
        Almacenar historia de cambios en clasificación.
        
        Args:
            classification_id: ID de la clasificación
            old_score: Puntuación anterior
            new_score: Puntuación nueva
            old_hot: Estado hot anterior
            new_hot: Estado hot nuevo
            change_magnitude: Magnitud del cambio (porcentaje)
        """
        # Determinar condición de mercado
        market_condition = "neutral"
        if change_magnitude > 10:
            market_condition = "bull"
        elif change_magnitude < -10:
            market_condition = "bear"
        
        # Determinar razón del cambio
        if old_hot != new_hot:
            reason = "Cambio de estado hot" + (" → hot" if new_hot else " → normal")
        else:
            reason = f"Cambio de puntuación ({change_magnitude:.1f}%)"
        
        insert_history_query = (
            insert(ClassificationHistory).values(
                classification_id=classification_id,
                previous_final_score=old_score,
                new_final_score=new_score,
                previous_hot_rating=old_hot,
                new_hot_rating=new_hot,
                capital_base=self.current_capital,
                market_condition=market_condition,
                change_magnitude=change_magnitude,
                change_reason=reason,
                change_date=datetime.now()
            )
        ), []
        
        await transcendental_db.execute_query(insert_history_query)
    
    async def _store_capital_scale_analysis(
        self,
        crypto_id: int,
        scores_by_capital: Dict[str, float],
        scale_sensitivity: float,
        limiting_factor: str,
        saturation_point: float
    ) -> None:
        """
        Almacenar análisis de efecto de escala.
        
        Args:
            crypto_id: ID de la criptomoneda
            scores_by_capital: Puntuaciones por nivel de capital
            scale_sensitivity: Sensibilidad a la escala
            limiting_factor: Factor limitante
            saturation_point: Punto de saturación
        """
        # Verificar si ya existe un análisis reciente (< 7 días)
        check_recent_query = (
            select(CapitalScaleEffect)
            .where(
                and_(
                    CapitalScaleEffect.cryptocurrency_id == crypto_id,
                    CapitalScaleEffect.analysis_date > datetime.now() - timedelta(days=7)
                )
            )
            .limit(1)
        ), []
        
        recent = await transcendental_db.execute_query(check_recent_query)
        
        if recent:
            # Existe análisis reciente, actualizarlo
            analysis_id = recent[0].id
            
            update_query = (
                update(CapitalScaleEffect)
                .where(CapitalScaleEffect.id == analysis_id)
                .values(
                    score_10k=scores_by_capital.get("10000", 0.0),
                    score_100k=scores_by_capital.get("100000", 0.0),
                    score_1m=scores_by_capital.get("1000000", 0.0),
                    score_10m=scores_by_capital.get("10000000", 0.0),
                    scale_sensitivity=scale_sensitivity,
                    max_effective_capital=saturation_point,
                    limiting_factor=limiting_factor,
                    saturation_point=saturation_point,
                    analysis_date=datetime.now()
                )
            ), []
            
            await transcendental_db.execute_query(update_query)
        else:
            # No existe, crear nuevo análisis
            insert_query = (
                insert(CapitalScaleEffect).values(
                    cryptocurrency_id=crypto_id,
                    score_10k=scores_by_capital.get("10000", 0.0),
                    score_100k=scores_by_capital.get("100000", 0.0),
                    score_1m=scores_by_capital.get("1000000", 0.0),
                    score_10m=scores_by_capital.get("10000000", 0.0),
                    scale_sensitivity=scale_sensitivity,
                    max_effective_capital=saturation_point,
                    limiting_factor=limiting_factor,
                    saturation_point=saturation_point,
                    analysis_date=datetime.now()
                )
            ), []
            
            await transcendental_db.execute_query(insert_query)


# Instancia global del clasificador
classifier = TranscendentalCryptoClassifier()

async def initialize_classifier(initial_capital: float, config: Dict[str, Any] = None) -> None:
    """
    Inicializar el clasificador transcendental con configuración específica.
    
    Args:
        initial_capital: Capital inicial para el clasificador
        config: Configuración adicional
    """
    global classifier
    
    if config is None:
        config = {}
    
    # Extraer configuración
    confidence_threshold = config.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
    exchanges = config.get('exchanges', DEFAULT_EXCHANGES)
    
    # Reinicializar el clasificador con nuevos parámetros
    classifier = TranscendentalCryptoClassifier(
        initial_capital=initial_capital,
        exchanges=exchanges,
        confidence_threshold=confidence_threshold
    )
    
    logger.info(f"Clasificador transcendental inicializado con capital={initial_capital}, "
                f"{len(exchanges)} exchanges, threshold={confidence_threshold}")
    
    # Realizar clasificación inicial si es necesario
    if config.get('initial_classification', False):
        await classifier.classify_all()