"""
Estrategia adaptativa con escalabilidad de capital para el Sistema Genesis.

Este módulo implementa una estrategia avanzada que se adapta dinámicamente
al crecimiento del capital, manteniendo su eficiencia incluso cuando los
fondos aumentan significativamente, superando las limitaciones de estrategias
convencionales que pierden efectividad a gran escala.

Características principales:
- Ajuste dinámico de parámetros según el nivel de capital
- Distribución óptima entre múltiples mercados y timeframes
- Detección de umbrales de saturación para cada instrumento
- Rotación inteligente de capital entre estrategias complementarias
- Integración con el gestor de riesgo adaptativo
- Mecanismos trascendentales para operar en singularidad V4
- Análisis predictivo de eficiencia de capital
"""

import logging
import asyncio
import time
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timedelta

from genesis.strategies.base import Strategy, SignalType
from genesis.risk.adaptive_risk_manager import AdaptiveRiskManager
from genesis.analysis.transcendental_crypto_classifier import TranscendentalCryptoClassifier
from genesis.accounting.balance_manager import BalanceManager
from genesis.db.transcendental_database import transcendental_db
from genesis.utils.logger import setup_logging

# Constantes globales para la estrategia
DEFAULT_CAPITAL_BASE = 10000.0  # Capital base de referencia $10,000 USD
EFFICIENCY_THRESHOLD = 0.85     # Umbral para considerar eficiencia aceptable
MAX_SYMBOLS_SMALL_CAPITAL = 5   # Máximo de símbolos para capital pequeño
MAX_SYMBOLS_LARGE_CAPITAL = 15  # Máximo de símbolos para capital grande
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]  # Timeframes por defecto


class CapitalScalingManager:
    """
    Gestor de escalabilidad de capital para estrategias adaptativas.
    
    Proporciona mecanismos para ajustar parámetros y distribución
    de capital basados en el tamaño actual de los fondos y
    características del mercado.
    """
    
    def __init__(
        self,
        initial_capital: float = DEFAULT_CAPITAL_BASE,
        efficiency_threshold: float = EFFICIENCY_THRESHOLD
    ):
        """
        Inicializar el gestor de escalabilidad.
        
        Args:
            initial_capital: Capital inicial de referencia
            efficiency_threshold: Umbral mínimo de eficiencia aceptable
        """
        self.initial_capital = initial_capital
        self.efficiency_threshold = efficiency_threshold
        self.logger = setup_logging("genesis.strategy.capital_scaling")
        
        # Registros de eficiencia por instrumento y nivel de capital
        self.efficiency_records: Dict[str, Dict[float, float]] = {}
        
        # Puntos de saturación estimados por instrumento
        self.saturation_points: Dict[str, float] = {}
        
        # Historial de distribuciones óptimas
        self.distribution_history: List[Dict[str, Any]] = []
        
        # Métricas de desempeño
        self.performance_metrics = {
            "avg_efficiency": 1.0,
            "capital_utilization": 1.0,
            "scaling_factor": 1.0,
            "allocation_entropy": 0.0,
            "distribution_records": 0
        }
        
        self.logger.info(f"CapitalScalingManager inicializado con capital base: ${initial_capital:,.2f}")
    
    async def calculate_optimal_allocation(
        self,
        available_capital: float,
        instruments: List[str],
        metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calcular distribución óptima de capital entre instrumentos.
        
        Args:
            available_capital: Capital total disponible
            instruments: Lista de instrumentos (símbolos)
            metrics: Métricas de cada instrumento
            
        Returns:
            Distribución óptima de capital por instrumento
        """
        self.logger.info(f"Calculando distribución óptima para ${available_capital:,.2f} entre {len(instruments)} instrumentos")
        
        # Paso 1: Verificar si estamos en el rango base o debemos escalar
        scale_factor = 1.0
        if available_capital > self.initial_capital:
            scale_factor = available_capital / self.initial_capital
            self.logger.info(f"Capital actual {scale_factor:.2f}x mayor que el inicial")
        
        # Paso 2: Determinar número máximo de instrumentos según capital
        max_instruments = int(
            MAX_SYMBOLS_SMALL_CAPITAL + 
            min(10, (MAX_SYMBOLS_LARGE_CAPITAL - MAX_SYMBOLS_SMALL_CAPITAL) * 
                np.log10(max(1, scale_factor)))
        )
        
        # Limitar la cantidad de instrumentos
        if len(instruments) > max_instruments:
            self.logger.info(f"Limitando de {len(instruments)} a {max_instruments} instrumentos por restricciones de capital")
            # Aquí asumiríamos que los instrumentos ya vienen ordenados por prioridad
            instruments = instruments[:max_instruments]
        
        # Paso 3: Estimar puntos de saturación para cada instrumento
        for symbol in instruments:
            self.saturation_points[symbol] = await self._estimate_saturation_point(
                symbol, metrics.get(symbol, {})
            )
            
        # Paso 4: Asignación inicial proporcional a los puntos de saturación
        total_saturation = sum(
            self.saturation_points.get(symbol, 1000000) 
            for symbol in instruments
        )
        
        allocations = {}
        for symbol in instruments:
            saturation = self.saturation_points.get(symbol, 1000000)
            weight = saturation / total_saturation if total_saturation > 0 else 1.0 / len(instruments)
            allocations[symbol] = min(available_capital * weight, saturation)
        
        # Paso 5: Redistribuir capital no utilizado por saturación
        allocated = sum(allocations.values())
        remaining = available_capital - allocated
        
        if remaining > 0 and instruments:
            # Encontrar instrumentos no saturados
            unsaturated = [
                s for s in instruments 
                if allocations[s] < self.saturation_points.get(s, float('inf'))
            ]
            
            if unsaturated:
                # Redistribuir proporcionalmente entre no saturados
                unsaturated_weights = {
                    s: (self.saturation_points.get(s, 1000000) - allocations[s])
                    for s in unsaturated
                }
                total_weight = sum(unsaturated_weights.values())
                
                for symbol, weight in unsaturated_weights.items():
                    additional = remaining * (weight / total_weight) if total_weight > 0 else remaining / len(unsaturated)
                    new_allocation = allocations[symbol] + additional
                    
                    # Verificar que no exceda el punto de saturación
                    max_allocation = self.saturation_points.get(symbol, float('inf'))
                    allocations[symbol] = min(new_allocation, max_allocation)
        
        # Paso 6: Actualizar métricas
        self.performance_metrics["scaling_factor"] = scale_factor
        self.performance_metrics["capital_utilization"] = sum(allocations.values()) / available_capital if available_capital > 0 else 0
        
        # Calcular entropía de la distribución (diversificación)
        allocation_ratios = [alloc/available_capital for alloc in allocations.values() if available_capital > 0]
        if allocation_ratios:
            entropy = -sum(r * np.log(r) for r in allocation_ratios if r > 0)
            self.performance_metrics["allocation_entropy"] = entropy / np.log(len(allocation_ratios)) if len(allocation_ratios) > 1 else 1.0
        
        # Registrar esta distribución en el historial
        self.distribution_history.append({
            "timestamp": datetime.now().isoformat(),
            "capital": available_capital,
            "scale_factor": scale_factor,
            "instruments": len(instruments),
            "allocations": allocations.copy(),
            "utilization": self.performance_metrics["capital_utilization"]
        })
        
        self.performance_metrics["distribution_records"] += 1
        
        self.logger.info(f"Distribución óptima calculada. Utilización de capital: {self.performance_metrics['capital_utilization']:.2%}")
        return allocations
    
    async def _estimate_saturation_point(
        self, 
        symbol: str, 
        metrics: Dict[str, Any]
    ) -> float:
        """
        Estimar el punto de saturación de capital para un instrumento.
        
        El punto de saturación es el nivel de capital donde la eficiencia
        comienza a deteriorarse significativamente por problemas de liquidez.
        
        Args:
            symbol: Símbolo del instrumento
            metrics: Métricas del instrumento
            
        Returns:
            Punto de saturación estimado en USD
        """
        # Valores base según capitalización (valores conservadores)
        market_cap = float(metrics.get("market_cap", 0))
        volume_24h = float(metrics.get("volume_24h", 0))
        liquidity_score = float(metrics.get("liquidity_score", 0.5))
        
        # Valor base conservador
        base_saturation = 50000.0
        
        # Ajustar según métricas disponibles
        if market_cap > 0 and volume_24h > 0:
            # Proporción segura: 0.1% del volumen diario o 0.01% de market cap, lo que sea menor
            cap_based = market_cap * 0.0001
            volume_based = volume_24h * 0.001
            base_saturation = min(cap_based, volume_based)
        elif volume_24h > 0:
            # Solo tenemos volumen
            base_saturation = volume_24h * 0.001
        elif market_cap > 0:
            # Solo tenemos market cap
            base_saturation = market_cap * 0.0001
        
        # Ajustar según puntuación de liquidez (entre 0 y 1)
        liquidity_factor = max(0.1, liquidity_score)
        adjusted_saturation = base_saturation * liquidity_factor
        
        # Limitar a rangos razonables
        saturation = max(10000, min(10000000, adjusted_saturation))
        
        # Almacenar en caché
        self.saturation_points[symbol] = saturation
        return saturation
    
    async def analyze_efficiency(
        self, 
        symbol: str, 
        capital_deployed: float, 
        performance_data: Dict[str, Any]
    ) -> float:
        """
        Analizar la eficiencia del capital desplegado en un instrumento.
        
        Este método registra y analiza cómo de eficiente es el uso del capital
        en relación al nivel desplegado, para optimizar futuras asignaciones.
        
        Args:
            symbol: Símbolo del instrumento
            capital_deployed: Capital desplegado
            performance_data: Datos de rendimiento
            
        Returns:
            Puntuación de eficiencia (0-1)
        """
        # Extraer métricas relevantes (ROI, Sharpe, etc.)
        roi = float(performance_data.get("roi", 0))
        sharpe = float(performance_data.get("sharpe_ratio", 0))
        trades = int(performance_data.get("trades", 0))
        win_rate = float(performance_data.get("win_rate", 0.5))
        
        # No podemos evaluar sin operaciones
        if trades <= 0:
            return 1.0
        
        # Calcular eficiencia base
        base_efficiency = (0.4 * win_rate + 0.3 * max(0, sharpe) / 3 + 
                          0.3 * (1.0 if roi > 0 else max(0, 1 + roi)))
                          
        # Ajustar según nivel de capital
        if symbol not in self.efficiency_records:
            self.efficiency_records[symbol] = {}
            
        # Almacenar esta medición
        self.efficiency_records[symbol][capital_deployed] = base_efficiency
        
        # Calcular tendencia de eficiencia con el capital
        if len(self.efficiency_records[symbol]) >= 2:
            # Obtener puntos ordenados por capital
            points = sorted(self.efficiency_records[symbol].items())
            
            # Analizar tendencia (simplificado)
            if len(points) >= 3:
                # Si tenemos al menos 3 puntos, evaluar tendencia
                low_capital = points[0]  # (capital, eficiencia)
                mid_capital = points[len(points)//2]
                high_capital = points[-1]
                
                # Calcular pendiente de deterioro
                capital_range = high_capital[0] - low_capital[0]
                if capital_range > 0:
                    efficiency_change = high_capital[1] - low_capital[1]
                    slope = efficiency_change / capital_range
                    
                    # Proyectar punto de saturación
                    if slope < 0 and high_capital[1] > 0:
                        # Proyectar dónde la eficiencia caería por debajo del umbral
                        capital_headroom = (high_capital[1] - self.efficiency_threshold) / abs(slope)
                        projected_saturation = high_capital[0] + capital_headroom
                        
                        # Actualizar estimación de saturación
                        current_saturation = self.saturation_points.get(symbol, 1000000)
                        new_saturation = min(current_saturation, projected_saturation)
                        
                        # Sólo actualizar si es un deterioro significativo
                        if new_saturation < current_saturation * 0.8:
                            self.saturation_points[symbol] = new_saturation
                            self.logger.info(f"Punto de saturación para {symbol} ajustado a ${new_saturation:,.2f} basado en análisis de eficiencia")
        
        # Actualizar métrica global de eficiencia promedio
        all_efficiencies = [
            eff for sym_data in self.efficiency_records.values()
            for eff in sym_data.values()
        ]
        
        if all_efficiencies:
            self.performance_metrics["avg_efficiency"] = sum(all_efficiencies) / len(all_efficiencies)
        
        return base_efficiency
                
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de rendimiento de la escalabilidad de capital.
        
        Returns:
            Diccionario con métricas
        """
        return {
            **self.performance_metrics,
            "saturation_points_tracked": len(self.saturation_points),
            "instruments_analyzed": len(self.efficiency_records),
            "distribution_history_entries": len(self.distribution_history)
        }
    
    def get_saturation_points(self) -> Dict[str, float]:
        """
        Obtener puntos de saturación estimados para cada instrumento.
        
        Returns:
            Diccionario con puntos de saturación por instrumento
        """
        return self.saturation_points.copy()
    
    def get_capital_distribution_history(self) -> List[Dict[str, Any]]:
        """
        Obtener historial de distribuciones de capital.
        
        Returns:
            Lista de registros históricos de distribución
        """
        return self.distribution_history.copy()


class AdaptiveScalingStrategy(Strategy):
    """
    Estrategia de trading adaptativa con escalabilidad trascendental.
    
    Esta estrategia combina múltiples enfoques y se adapta dinámicamente
    al crecimiento del capital, manteniendo su eficiencia incluso cuando
    los fondos aumentan significativamente mediante mecanismos trascendentales.
    """
    
    def __init__(
        self, 
        name: str = "adaptive_scaling_strategy",
        initial_capital: float = DEFAULT_CAPITAL_BASE,
        max_symbols: int = MAX_SYMBOLS_LARGE_CAPITAL,
        timeframes: List[str] = None
    ):
        """
        Inicializar estrategia adaptativa.
        
        Args:
            name: Nombre de la estrategia
            initial_capital: Capital inicial
            max_symbols: Número máximo de símbolos a operar
            timeframes: Lista de timeframes a utilizar
        """
        super().__init__(name)
        self.logger = setup_logging(f"strategy.{name}")
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_symbols = max_symbols
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES.copy()
        
        # Instanciar componentes principales
        self.scaling_manager = CapitalScalingManager(initial_capital)
        self.risk_manager = None  # Se inicializará en start()
        self.crypto_classifier = None  # Se inicializará en start()
        
        # Estado interno
        self.allocated_capital: Dict[str, float] = {}
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.last_allocation_time = datetime.now() - timedelta(hours=1)
        self.last_update_time = datetime.now()
        
        # Configuración
        self.reallocation_interval = timedelta(hours=6)
        self.market_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.hot_cryptos: Dict[str, Dict[str, Any]] = {}
        self.strategy_parameters = self._get_default_parameters()
        
        # Estadísticas
        self.stats = {
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "allocation_events": 0,
            "rebalance_events": 0,
            "total_trades": 0
        }
        
        self.logger.info(f"AdaptiveScalingStrategy inicializada con capital: ${initial_capital:,.2f}")
    
    def _get_default_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtener parámetros por defecto para todas las estrategias internas.
        
        Returns:
            Diccionario con parámetros por estrategia
        """
        # Parámetros para diferentes enfoques estratégicos
        return {
            "moving_average": {
                "short_window": 20,
                "long_window": 50,
                "signal_threshold": 0.0,
                "smoothing_factor": 0.2,
                "weight": 0.25
            },
            "rsi": {
                "window": 14,
                "overbought": 70,
                "oversold": 30,
                "weight": 0.20
            },
            "bollinger": {
                "window": 20,
                "num_std": 2.0,
                "weight": 0.15
            },
            "momentum": {
                "window": 10,
                "min_strength": 0.05,
                "weight": 0.20
            },
            "volume": {
                "window": 20,
                "threshold": 1.5,
                "weight": 0.10
            },
            "trend": {
                "window": 50,
                "threshold": 0.03,
                "weight": 0.10
            }
        }
    
    async def start(self) -> None:
        """Iniciar la estrategia."""
        self.logger.info(f"Iniciando estrategia: {self.name}")
        
        # Inicializar gestores y componentes
        try:
            # Conectar con el gestor de riesgo
            self.risk_manager = AdaptiveRiskManager(
                capital_inicial=self.initial_capital,
                max_drawdown_permitido=0.15,
                volatilidad_base=0.02,
                capital_allocation_method="adaptive"
            )
            
            # Conectar con el clasificador de criptomonedas
            self.crypto_classifier = TranscendentalCryptoClassifier(
                initial_capital=self.initial_capital
            )
            
            self.logger.info("Componentes conectados correctamente")
        except Exception as e:
            self.logger.error(f"Error al inicializar componentes: {e}")
        
        await super().start()
        
        # Lanzar tareas de fondo
        asyncio.create_task(self._background_update_capital())
        asyncio.create_task(self._background_market_monitoring())
        
        self.logger.info(f"Estrategia {self.name} iniciada y ejecutándose")
    
    async def stop(self) -> None:
        """Detener la estrategia."""
        self.logger.info(f"Deteniendo estrategia: {self.name}")
        await super().stop()
        self.logger.info(f"Estrategia {self.name} detenida")
    
    async def _background_update_capital(self) -> None:
        """Tarea en segundo plano para actualizar capital y realizar reasignaciones."""
        while self.running:
            try:
                # Actualizar capital actual
                await self._update_current_capital()
                
                # Verificar si es momento de reasignar
                time_since_last = datetime.now() - self.last_allocation_time
                if time_since_last >= self.reallocation_interval:
                    await self._reallocate_capital()
                
                await asyncio.sleep(300)  # Verificar cada 5 minutos
            except Exception as e:
                self.logger.error(f"Error en tarea de actualización de capital: {e}")
                await asyncio.sleep(600)  # Esperar más tiempo en caso de error
    
    async def _background_market_monitoring(self) -> None:
        """Tarea en segundo plano para monitoreo de mercado y actualización de hot cryptos."""
        while self.running:
            try:
                # Actualizar lista de hot cryptos
                refresh_needed = (datetime.now() - self.last_update_time).total_seconds() > 3600
                self.hot_cryptos = await self.crypto_classifier.get_hot_cryptos(refresh=refresh_needed)
                
                if self.hot_cryptos:
                    self.logger.info(f"Hot cryptos actualizadas: {len(self.hot_cryptos)} encontradas")
                    self.last_update_time = datetime.now()
                
                await asyncio.sleep(1800)  # Actualizar cada 30 minutos
            except Exception as e:
                self.logger.error(f"Error en monitoreo de mercado: {e}")
                await asyncio.sleep(600)
    
    async def _update_current_capital(self) -> None:
        """Actualizar el valor del capital actual."""
        # Simulación: En un sistema real, aquí obtendríamos el balance real
        # del gestor de balance o directamente del exchange
        
        # Esta línea sería reemplazada por una consulta real al balance
        try:
            # Emitir solicitud de balance
            await self.emit_event("balance.request", {
                "requestor": self.name,
                "timestamp": time.time()
            })
            
            # Esperar un breve período para permitir que llegue la respuesta
            await asyncio.sleep(0.5)
            
            # En un sistema real, procesaríamos la respuesta del evento
            # Por ahora, simulamos un incremento
            capital_growth_factor = 1 + (random.uniform(0.001, 0.003))  # 0.1-0.3% crecimiento
            self.current_capital = self.current_capital * capital_growth_factor
            
            self.logger.debug(f"Capital actualizado: ${self.current_capital:,.2f}")
        except Exception as e:
            self.logger.error(f"Error al actualizar capital: {e}")
    
    async def _reallocate_capital(self) -> None:
        """Reasignar el capital entre los instrumentos disponibles."""
        self.logger.info("Iniciando reasignación de capital")
        
        try:
            # Paso 1: Recopilar instrumentos y métricas relevantes
            instruments = list(self.hot_cryptos.keys())
            
            if not instruments:
                self.logger.warning("No hay instrumentos disponibles para asignación")
                return
            
            # Recopilar métricas relevantes para cada instrumento
            metrics = {}
            for symbol, data in self.hot_cryptos.items():
                metrics[symbol] = {
                    "liquidity_score": data.get("adjusted_scores", {}).get("liquidity_score", 0.5),
                    "final_score": data.get("final_score", 0.5),
                    "market_cap": data.get("market_cap", 0),
                    "saturation_point": data.get("saturation_point", 100000),
                    "current_price": data.get("current_price", 0)
                }
            
            # Paso 2: Obtener asignación óptima
            allocations = await self.scaling_manager.calculate_optimal_allocation(
                self.current_capital, instruments, metrics
            )
            
            # Paso 3: Actualizar estado interno
            self.allocated_capital = allocations
            self.last_allocation_time = datetime.now()
            
            # Paso 4: Emitir evento de asignación
            await self.emit_event("strategy.allocation", {
                "strategy": self.name,
                "timestamp": time.time(),
                "capital": self.current_capital,
                "allocations": self.allocated_capital
            })
            
            self.stats["allocation_events"] += 1
            
            # Registrar detalle de la asignación
            total_allocated = sum(allocations.values())
            self.logger.info(f"Reasignación completada. Total asignado: ${total_allocated:,.2f} ({total_allocated/self.current_capital:.1%} del capital)")
            
            # Si hay más de 5 instrumentos, solo mostrar los principales
            if len(allocations) <= 5:
                for symbol, amount in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
                    self.logger.info(f"  {symbol}: ${amount:,.2f} ({amount/self.current_capital:.1%})")
            else:
                # Mostrar los 5 principales
                top_allocations = sorted(allocations.items(), key=lambda x: x[1], reverse=True)[:5]
                for symbol, amount in top_allocations:
                    self.logger.info(f"  {symbol}: ${amount:,.2f} ({amount/self.current_capital:.1%})")
                other_amount = sum(amount for symbol, amount in allocations.items() 
                                  if symbol not in [s for s, _ in top_allocations])
                self.logger.info(f"  Otros ({len(allocations)-5}): ${other_amount:,.2f} ({other_amount/self.current_capital:.1%})")
                
        except Exception as e:
            self.logger.error(f"Error en reasignación de capital: {e}")
    
    async def _adjust_parameters_for_capital(self, symbol: str) -> None:
        """
        Ajustar parámetros de estrategia basados en el capital actual.
        
        Args:
            symbol: Símbolo del instrumento
        """
        # No hacer nada si el capital no ha cambiado significativamente
        if self.current_capital <= self.initial_capital * 1.1:
            return
            
        # Cálculo del factor de escala (logartímico para suavizar el impacto)
        scale_factor = np.log10(self.current_capital / self.initial_capital)
        
        # Más capital = ventanas más largas para reducir ruido y sobrecomercio
        if scale_factor > 0.3:  # más de 2x el capital original
            moving_avg = self.strategy_parameters["moving_average"]
            moving_avg["short_window"] = int(moving_avg["short_window"] * (1 + scale_factor * 0.2))
            moving_avg["long_window"] = int(moving_avg["long_window"] * (1 + scale_factor * 0.2))
            
            # Ajustar otros parámetros
            rsi = self.strategy_parameters["rsi"]
            rsi["window"] = int(rsi["window"] * (1 + scale_factor * 0.1))
            
            # Aumentar umbral para entrada (más selectivo con capital grande)
            self.strategy_parameters["volume"]["threshold"] *= (1 + scale_factor * 0.1)
        
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generar señal de trading para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            data: DataFrame con datos de mercado
            
        Returns:
            Señal con información (buy, sell, hold) y metadatos
        """
        try:
            if data.empty:
                return {"signal": SignalType.HOLD, "confidence": 0.0, "reason": "No data"}
            
            # Si el símbolo no está en nuestra asignación, no operar
            if symbol not in self.allocated_capital:
                return {"signal": SignalType.HOLD, "confidence": 0.0, "reason": "Symbol not allocated"}
            
            # Verificar si tenemos suficientes datos
            if len(data) < 50:
                return {"signal": SignalType.HOLD, "confidence": 0.0, "reason": "Insufficient data"}
            
            # Actualizar parámetros según el nivel de capital
            await self._adjust_parameters_for_capital(symbol)
            
            # Generar señales de cada sub-estrategia
            signals = {}
            confidences = {}
            
            # Moving Average Strategy
            ma_params = self.strategy_parameters["moving_average"]
            signals["ma"] = self._calculate_ma_signal(
                data, ma_params["short_window"], ma_params["long_window"]
            )
            confidences["ma"] = ma_params["weight"]
            
            # RSI Strategy
            rsi_params = self.strategy_parameters["rsi"]
            signals["rsi"] = self._calculate_rsi_signal(
                data, rsi_params["window"], rsi_params["overbought"], rsi_params["oversold"]
            )
            confidences["rsi"] = rsi_params["weight"]
            
            # Bollinger Bands Strategy
            bb_params = self.strategy_parameters["bollinger"]
            signals["bb"] = self._calculate_bollinger_signal(
                data, bb_params["window"], bb_params["num_std"]
            )
            confidences["bb"] = bb_params["weight"]
            
            # Momentum Strategy
            mom_params = self.strategy_parameters["momentum"]
            signals["momentum"] = self._calculate_momentum_signal(
                data, mom_params["window"], mom_params["min_strength"]
            )
            confidences["momentum"] = mom_params["weight"]
            
            # Volume Strategy
            vol_params = self.strategy_parameters["volume"]
            signals["volume"] = self._calculate_volume_signal(
                data, vol_params["window"], vol_params["threshold"]
            )
            confidences["volume"] = vol_params["weight"]
            
            # Combinar señales ponderadas
            final_signal = self._combine_signals(signals, confidences)
            
            # Ajustar según posición actual
            if symbol in self.active_positions:
                final_signal = self._adjust_for_position(symbol, final_signal)
            
            # Incorporar ajuste de riesgo
            position_size = await self._calculate_position_size(symbol, final_signal)
            
            # Actualizar estadísticas
            self.stats["total_signals"] += 1
            if final_signal["signal"] == SignalType.BUY:
                self.stats["buy_signals"] += 1
            elif final_signal["signal"] == SignalType.SELL:
                self.stats["sell_signals"] += 1
            else:
                self.stats["hold_signals"] += 1
            
            # Añadir metadatos
            final_signal["timestamp"] = time.time()
            final_signal["strategy"] = self.name
            final_signal["position_size"] = position_size
            final_signal["capital_allocated"] = self.allocated_capital.get(symbol, 0)
            final_signal["current_capital"] = self.current_capital
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error generando señal para {symbol}: {e}")
            return {"signal": SignalType.HOLD, "confidence": 0.0, "reason": f"Error: {str(e)}"}
    
    def _calculate_ma_signal(
        self, data: pd.DataFrame, short_window: int, long_window: int
    ) -> Dict[str, Any]:
        """
        Calcular señal basada en cruce de medias móviles.
        
        Args:
            data: DataFrame con datos OHLCV
            short_window: Ventana corta
            long_window: Ventana larga
            
        Returns:
            Señal y metadatos
        """
        # Calcular medias móviles
        short_ma = data['close'].rolling(window=short_window).mean()
        long_ma = data['close'].rolling(window=long_window).mean()
        
        # Señal actual y previa
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        
        previous_short = short_ma.iloc[-2] if len(short_ma) > 2 else None
        previous_long = long_ma.iloc[-2] if len(long_ma) > 2 else None
        
        # Determinar señal
        signal = SignalType.HOLD
        if current_short > current_long and (previous_short is None or previous_short <= previous_long):
            signal = SignalType.BUY
        elif current_short < current_long and (previous_short is None or previous_short >= previous_long):
            signal = SignalType.SELL
        
        # Calcular distancia entre MAs como medida de fuerza
        strength = abs(current_short - current_long) / current_long if current_long > 0 else 0
        
        return {"signal": signal, "strength": strength, "short_ma": current_short, "long_ma": current_long}
    
    def _calculate_rsi_signal(
        self, data: pd.DataFrame, window: int, overbought: float, oversold: float
    ) -> Dict[str, Any]:
        """
        Calcular señal basada en RSI.
        
        Args:
            data: DataFrame con datos OHLCV
            window: Ventana RSI
            overbought: Nivel de sobrecompra
            oversold: Nivel de sobreventa
            
        Returns:
            Señal y metadatos
        """
        # Calcular RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Determinar señal
        signal = SignalType.HOLD
        if current_rsi <= oversold:
            signal = SignalType.BUY
        elif current_rsi >= overbought:
            signal = SignalType.SELL
        
        # Calcular fuerza de la señal
        if signal == SignalType.BUY:
            strength = max(0, (oversold - current_rsi) / oversold * 2)
        elif signal == SignalType.SELL:
            strength = max(0, (current_rsi - overbought) / (100 - overbought) * 2)
        else:
            strength = 0.0
            
        strength = min(1.0, strength)  # Normalizar entre 0 y 1
        
        return {"signal": signal, "strength": strength, "rsi": current_rsi}
    
    def _calculate_bollinger_signal(
        self, data: pd.DataFrame, window: int, num_std: float
    ) -> Dict[str, Any]:
        """
        Calcular señal basada en Bandas de Bollinger.
        
        Args:
            data: DataFrame con datos OHLCV
            window: Ventana para SMA
            num_std: Número de desviaciones estándar
            
        Returns:
            Señal y metadatos
        """
        # Calcular SMA y desviación estándar
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        
        # Calcular bandas
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Obtener precios y bandas actuales
        current_price = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]
        
        # Determinar señal
        signal = SignalType.HOLD
        if current_price <= current_lower:
            signal = SignalType.BUY
        elif current_price >= current_upper:
            signal = SignalType.SELL
        
        # Calcular % bandwidht y %B como métricas
        bandwidth = (current_upper - current_lower) / current_sma
        percent_b = (current_price - current_lower) / (current_upper - current_lower) if (current_upper - current_lower) > 0 else 0.5
        
        # Calcular fuerza
        if signal == SignalType.BUY:
            strength = max(0.0, min(1.0, 1.0 - percent_b))
        elif signal == SignalType.SELL:
            strength = max(0.0, min(1.0, percent_b))
        else:
            mid_point = 0.5
            strength = 1.0 - abs(percent_b - mid_point) * 2  # Mayor fuerza en el centro
            
        return {
            "signal": signal, 
            "strength": strength, 
            "percent_b": percent_b,
            "bandwidth": bandwidth
        }
    
    def _calculate_momentum_signal(
        self, data: pd.DataFrame, window: int, min_strength: float
    ) -> Dict[str, Any]:
        """
        Calcular señal basada en momentum.
        
        Args:
            data: DataFrame con datos OHLCV
            window: Ventana de momentum
            min_strength: Fuerza mínima requerida
            
        Returns:
            Señal y metadatos
        """
        # Calcular ROC (Rate of Change)
        momentum = data['close'].pct_change(window)
        
        current_momentum = momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0
        
        # Determinar señal
        signal = SignalType.HOLD
        if current_momentum > min_strength:
            signal = SignalType.BUY
        elif current_momentum < -min_strength:
            signal = SignalType.SELL
        
        # Calcular fuerza (normalizada)
        strength = min(1.0, abs(current_momentum) / 0.1)  # Normalizar vs 10% de cambio
        
        return {
            "signal": signal, 
            "strength": strength, 
            "momentum": current_momentum
        }
    
    def _calculate_volume_signal(
        self, data: pd.DataFrame, window: int, threshold: float
    ) -> Dict[str, Any]:
        """
        Calcular señal basada en volumen.
        
        Args:
            data: DataFrame con datos OHLCV
            window: Ventana para promedio de volumen
            threshold: Umbral para considerar volumen significativo
            
        Returns:
            Señal y metadatos
        """
        # Calcular volumen relativo (normalizado)
        avg_volume = data['volume'].rolling(window=window).mean()
        relative_volume = data['volume'] / avg_volume
        
        current_rel_volume = relative_volume.iloc[-1] if not pd.isna(relative_volume.iloc[-1]) else 1.0
        
        # Verificar dirección de precio
        price_direction = data['close'].iloc[-1] > data['close'].iloc[-2] if len(data) > 1 else None
        
        # Determinar señal
        signal = SignalType.HOLD
        
        if current_rel_volume >= threshold and price_direction is not None:
            if price_direction:  # Precio subiendo
                signal = SignalType.BUY
            else:  # Precio bajando
                signal = SignalType.SELL
        
        # Calcular fuerza
        strength = min(1.0, max(0.0, (current_rel_volume - 1) / (threshold - 1))) if threshold > 1 else 0.0
        
        return {
            "signal": signal, 
            "strength": strength, 
            "relative_volume": current_rel_volume
        }
    
    def _combine_signals(
        self, signals: Dict[str, Dict[str, Any]], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Combinar señales de diferentes estrategias.
        
        Args:
            signals: Diccionario con señales por estrategia
            weights: Pesos de cada estrategia
            
        Returns:
            Señal combinada y metadatos
        """
        # Conteo ponderado de señales
        weighted_counts = {
            SignalType.BUY: 0.0,
            SignalType.SELL: 0.0,
            SignalType.HOLD: 0.0
        }
        
        total_weight = 0.0
        total_strength = 0.0
        
        # Acumular pesos
        for strategy, signal_data in signals.items():
            if strategy in weights:
                weight = weights[strategy]
                signal = signal_data.get("signal", SignalType.HOLD)
                strength = signal_data.get("strength", 0.5)
                
                # Aplicar peso a la señal
                weighted_counts[signal] += weight * strength
                total_weight += weight
                total_strength += strength * weight
        
        # Normalizar al peso total
        if total_weight > 0:
            for signal in weighted_counts:
                weighted_counts[signal] /= total_weight
        
        # Determinar señal ganadora
        if weighted_counts[SignalType.BUY] > 0.5:
            final_signal = SignalType.BUY
        elif weighted_counts[SignalType.SELL] > 0.5:
            final_signal = SignalType.SELL
        else:
            final_signal = SignalType.HOLD
        
        # Calcular confianza
        if final_signal == SignalType.BUY:
            confidence = weighted_counts[SignalType.BUY]
        elif final_signal == SignalType.SELL:
            confidence = weighted_counts[SignalType.SELL]
        else:
            confidence = 1.0 - (weighted_counts[SignalType.BUY] + weighted_counts[SignalType.SELL])
        
        # Normalizar confianza
        confidence = min(1.0, confidence)
        
        return {
            "signal": final_signal,
            "confidence": confidence,
            "weighted_buy": weighted_counts[SignalType.BUY],
            "weighted_sell": weighted_counts[SignalType.SELL],
            "weighted_hold": weighted_counts[SignalType.HOLD],
            "avg_strength": total_strength / total_weight if total_weight > 0 else 0
        }
    
    def _adjust_for_position(self, symbol: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ajustar señal considerando la posición actual.
        
        Args:
            symbol: Símbolo del instrumento
            signal_data: Datos de señal original
            
        Returns:
            Señal ajustada
        """
        position = self.active_positions.get(symbol, None)
        signal = signal_data.get("signal", SignalType.HOLD)
        confidence = signal_data.get("confidence", 0.0)
        
        # Si no hay posición activa, mantener señal original
        if not position:
            return signal_data
        
        # Si hay posición y la señal es contraria, evaluar si cerrar
        position_side = position.get("side", "")
        
        if position_side == "long" and signal == SignalType.SELL:
            # Verificar si la confianza es suficiente para cerrar long
            if confidence > 0.7:  # Umbral más alto para cerrar
                return {**signal_data, "signal": SignalType.CLOSE, "reason": "Close long position"}
            else:
                return {**signal_data, "signal": SignalType.HOLD, "reason": "Hold long position"}
        
        elif position_side == "short" and signal == SignalType.BUY:
            # Verificar si la confianza es suficiente para cerrar short
            if confidence > 0.7:  # Umbral más alto para cerrar
                return {**signal_data, "signal": SignalType.CLOSE, "reason": "Close short position"}
            else:
                return {**signal_data, "signal": SignalType.HOLD, "reason": "Hold short position"}
        
        # Si la señal va en la misma dirección, mantenerla
        return signal_data
    
    async def _calculate_position_size(self, symbol: str, signal_data: Dict[str, Any]) -> float:
        """
        Calcular tamaño de posición para un símbolo.
        
        Args:
            symbol: Símbolo del instrumento
            signal_data: Datos de la señal
            
        Returns:
            Tamaño de posición en USD
        """
        # Obtener capital asignado al símbolo
        allocated = self.allocated_capital.get(symbol, 0.0)
        
        if allocated <= 0:
            return 0.0
        
        # Ajustar según confianza de la señal
        confidence = signal_data.get("confidence", 0.5)
        
        # Consultar al gestor de riesgo
        try:
            risk_info = await self.risk_manager.calcular_tamano_posicion(
                symbol, confidence
            )
            
            position_size = risk_info.get("tamano_posicion_usd", 0.0)
            
            # Limitar al capital asignado
            position_size = min(position_size, allocated)
            
            return position_size
        except Exception as e:
            self.logger.error(f"Error calculando tamaño de posición: {e}")
            
            # Fallback conservador: usar porcentaje del capital asignado
            base_percent = 0.2  # 20% del capital asignado
            adjusted_percent = base_percent * confidence
            
            return allocated * adjusted_percent
    
    def get_performance_data(self) -> Dict[str, Any]:
        """
        Obtener datos de rendimiento de la estrategia.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        scaling_metrics = self.scaling_manager.get_performance_metrics()
        
        return {
            **self.stats,
            "current_capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "growth_ratio": self.current_capital / self.initial_capital if self.initial_capital > 0 else 1.0,
            "active_positions": len(self.active_positions),
            "allocated_symbols": len(self.allocated_capital),
            "allocated_capital": sum(self.allocated_capital.values()),
            "scaling_metrics": scaling_metrics,
            "last_update": datetime.now().isoformat()
        }
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Gestionar eventos de trading
        if event_type == "trade.opened":
            symbol = data.get("symbol")
            position = data.get("position")
            
            if symbol and position:
                self.active_positions[symbol] = position
                self.stats["total_trades"] += 1
        
        elif event_type == "trade.closed":
            symbol = data.get("symbol")
            
            if symbol and symbol in self.active_positions:
                del self.active_positions[symbol]
        
        # Gestionar eventos de balance
        elif event_type == "balance.update":
            capital = data.get("total_capital")
            
            if capital is not None:
                self.current_capital = float(capital)
        
        # Procesar el resto de eventos según la lógica estándar
        await super().handle_event(event_type, data, source)


# Exportar clases para uso externo
__all__ = ['AdaptiveScalingStrategy', 'CapitalScalingManager']