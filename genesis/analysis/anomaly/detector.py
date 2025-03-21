"""
Detector de anomalías para el sistema Genesis.

Este módulo proporciona algoritmos y funcionalidades para detectar anomalías
en los datos de mercado, como comportamientos inusuales de precio o volumen,
que pueden representar oportunidades de trading.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class AnomalyDetector(Component):
    """
    Detector de anomalías para datos de mercado.
    
    Este componente analiza datos de mercado para detectar comportamientos
    anómalos que pueden representar oportunidades de trading o señales
    de cambio en el régimen de mercado.
    """
    
    def __init__(self, name: str = "anomaly_detector"):
        """
        Inicializar el detector de anomalías.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Configuración
        self.sensitivity = 0.05  # Umbral de sensibilidad (0.01 = 1%)
        self.window_size = 100   # Tamaño de la ventana de análisis
        self.lookback_period = 50  # Período para establecer la línea base
        
        # Cache de datos por símbolo
        # symbol -> List[(timestamp, price, volume)]
        self.data_cache: Dict[str, List[Tuple[float, float, float]]] = {}
        
        # Resultados de detección por símbolo
        # symbol -> List[(timestamp, anomaly_type, score, details)]
        self.anomalies: Dict[str, List[Tuple[float, str, float, Dict[str, Any]]]] = {}
        
        # Límite de cache
        self.max_cache_size = 1000
        
        # Directorio para gráficos
        self.plots_dir = "data/plots/anomalies"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Inicializar módulos de detección
        self._init_detectors()
    
    def _init_detectors(self):
        """Inicializar los detectores de anomalías."""
        # Detector de isolation forest para price
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=self.sensitivity,
            random_state=42
        )
        
        # Detector de local outlier factor para volume
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.sensitivity
        )
        
        # DBSCAN para clustering de price-volume
        self.dbscan = DBSCAN(
            eps=0.3,
            min_samples=5
        )
    
    async def start(self) -> None:
        """Iniciar el detector de anomalías."""
        await super().start()
        self.logger.info("Detector de anomalías iniciado")
    
    async def stop(self) -> None:
        """Detener el detector de anomalías."""
        await super().stop()
        self.logger.info("Detector de anomalías detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Procesar actualizaciones de datos de mercado
        if event_type == "market.ticker_updated":
            await self._process_ticker_update(data)
        
        # Procesar solicitudes de análisis
        elif event_type == "analysis.detect_anomalies":
            symbol = data.get("symbol")
            if symbol:
                await self.detect_anomalies(symbol)
    
    async def _process_ticker_update(self, data: Dict[str, Any]) -> None:
        """
        Procesar actualización de ticker.
        
        Args:
            data: Datos del ticker
        """
        symbol = data.get("symbol")
        if not symbol:
            return
        
        # Extraer datos relevantes
        timestamp = data.get("timestamp", time.time())
        price = data.get("price", 0.0)
        volume = data.get("volume", 0.0)
        
        # Validar datos
        if price <= 0 or volume < 0:
            return
        
        # Actualizar cache
        if symbol not in self.data_cache:
            self.data_cache[symbol] = []
        
        self.data_cache[symbol].append((timestamp, price, volume))
        
        # Limitar tamaño de cache
        if len(self.data_cache[symbol]) > self.max_cache_size:
            self.data_cache[symbol] = self.data_cache[symbol][-self.max_cache_size:]
        
        # Detectar anomalías si hay suficientes datos
        if len(self.data_cache[symbol]) >= self.window_size:
            # Ejecutar detección cada 10 actualizaciones para reducir carga
            if len(self.data_cache[symbol]) % 10 == 0:
                await self.detect_anomalies(symbol)
    
    async def detect_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Detectar anomalías para un símbolo.
        
        Args:
            symbol: Símbolo a analizar
            
        Returns:
            Lista de anomalías detectadas
        """
        self.logger.info(f"Detectando anomalías para {symbol}")
        
        # Verificar si hay datos suficientes
        if symbol not in self.data_cache or len(self.data_cache[symbol]) < self.window_size:
            self.logger.warning(f"Datos insuficientes para {symbol}")
            return []
        
        # Convertir datos a arrays
        timestamps = []
        prices = []
        volumes = []
        
        for t, p, v in self.data_cache[symbol][-self.window_size:]:
            timestamps.append(t)
            prices.append(p)
            volumes.append(v)
        
        # Ejecutar métodos de detección
        anomalies = []
        
        # 1. Detectar anomalías de precio
        price_anomalies = await self._detect_price_anomalies(timestamps, prices, volumes)
        anomalies.extend(price_anomalies)
        
        # 2. Detectar anomalías de volumen
        volume_anomalies = await self._detect_volume_anomalies(timestamps, prices, volumes)
        anomalies.extend(volume_anomalies)
        
        # 3. Detectar patrones anómalos de precio-volumen
        pattern_anomalies = await self._detect_pattern_anomalies(timestamps, prices, volumes)
        anomalies.extend(pattern_anomalies)
        
        # Actualizar registro de anomalías
        if symbol not in self.anomalies:
            self.anomalies[symbol] = []
        
        # Convertir a formato interno y añadir
        for anomaly in anomalies:
            timestamp = anomaly.get("timestamp", time.time())
            anomaly_type = anomaly.get("type", "unknown")
            score = anomaly.get("score", 0.0)
            details = anomaly.get("details", {})
            
            self.anomalies[symbol].append((timestamp, anomaly_type, score, details))
        
        # Limitar tamaño del historial
        self.anomalies[symbol] = self.anomalies[symbol][-self.max_cache_size:]
        
        # Emitir evento para anomalías significativas (score > 0.7)
        for anomaly in anomalies:
            if anomaly.get("score", 0) > 0.7:
                await self.emit_event("analysis.anomaly_detected", {
                    "symbol": symbol,
                    "anomaly": anomaly
                })
                
                self.logger.info(f"Anomalía significativa detectada en {symbol}: {anomaly['type']}")
        
        # Generar gráfico si hay anomalías
        if anomalies:
            chart_path = await self._generate_anomaly_chart(symbol, timestamps, prices, volumes, anomalies)
            if chart_path:
                self.logger.info(f"Gráfico de anomalías generado: {chart_path}")
        
        return anomalies
    
    async def _detect_price_anomalies(
        self, 
        timestamps: List[float], 
        prices: List[float], 
        volumes: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Detectar anomalías en los precios.
        
        Args:
            timestamps: Lista de timestamps
            prices: Lista de precios
            volumes: Lista de volúmenes
            
        Returns:
            Lista de anomalías detectadas
        """
        anomalies = []
        
        # Extraer ventana de análisis
        window_data = np.array(prices).reshape(-1, 1)
        
        try:
            # 1. Detección estadística (Z-score)
            mean_price = np.mean(prices[:-1])  # Excluir último precio
            std_price = np.std(prices[:-1])
            
            if std_price > 0:
                z_scores = [(p - mean_price) / std_price for p in prices]
                
                # Último z-score
                last_z = z_scores[-1]
                
                # Considerar anomalía si |z| > 2.5
                if abs(last_z) > 2.5:
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "price_zscore",
                        "score": min(1.0, abs(last_z) / 5.0),  # Normalizar a [0,1]
                        "details": {
                            "z_score": last_z,
                            "price": prices[-1],
                            "mean_price": mean_price,
                            "std_price": std_price,
                            "direction": "up" if last_z > 0 else "down"
                        }
                    })
            
            # 2. Detección basada en machine learning (Isolation Forest)
            # Entrenar con datos históricos
            if len(prices) > self.lookback_period:
                train_data = window_data[:-1]  # Excluir último precio
                
                # Ajustar modelo
                self.isolation_forest.fit(train_data)
                
                # Predecir anomalías (menor score = más anómalo)
                anomaly_scores = self.isolation_forest.decision_function(window_data)
                
                # El último punto es anómalo?
                last_score = anomaly_scores[-1]
                
                # Considerar anomalía si score < 0
                if last_score < 0:
                    normalized_score = min(1.0, abs(last_score) / 0.5)  # Normalizar a [0,1]
                    
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "price_isolation_forest",
                        "score": normalized_score,
                        "details": {
                            "anomaly_score": last_score,
                            "price": prices[-1],
                            "normalized_score": normalized_score
                        }
                    })
            
            # 3. Detección de saltos bruscos (price gaps)
            price_changes = np.diff(prices)
            if len(price_changes) > 0:
                last_change = price_changes[-1]
                avg_change = np.mean(np.abs(price_changes[:-1]))
                
                if avg_change > 0 and abs(last_change) > 3 * avg_change:
                    # Salto significativo (> 3x el cambio promedio)
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "price_gap",
                        "score": min(1.0, abs(last_change) / (5 * avg_change)),
                        "details": {
                            "price_change": last_change,
                            "avg_change": avg_change,
                            "ratio": abs(last_change) / avg_change if avg_change > 0 else 0,
                            "direction": "up" if last_change > 0 else "down"
                        }
                    })
        
        except Exception as e:
            self.logger.error(f"Error en detección de anomalías de precio: {e}")
        
        return anomalies
    
    async def _detect_volume_anomalies(
        self, 
        timestamps: List[float], 
        prices: List[float], 
        volumes: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Detectar anomalías en el volumen.
        
        Args:
            timestamps: Lista de timestamps
            prices: Lista de precios
            volumes: Lista de volúmenes
            
        Returns:
            Lista de anomalías detectadas
        """
        anomalies = []
        
        # Extraer ventana de análisis
        window_data = np.array(volumes).reshape(-1, 1)
        
        try:
            # 1. Detección estadística (Z-score)
            mean_volume = np.mean(volumes[:-1])  # Excluir último volumen
            std_volume = np.std(volumes[:-1])
            
            if std_volume > 0:
                z_scores = [(v - mean_volume) / std_volume for v in volumes]
                
                # Último z-score
                last_z = z_scores[-1]
                
                # Considerar anomalía si |z| > 3 (más permisivo con volumen)
                if abs(last_z) > 3:
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "volume_zscore",
                        "score": min(1.0, abs(last_z) / 6.0),  # Normalizar a [0,1]
                        "details": {
                            "z_score": last_z,
                            "volume": volumes[-1],
                            "mean_volume": mean_volume,
                            "std_volume": std_volume,
                            "direction": "up" if last_z > 0 else "down"
                        }
                    })
            
            # 2. Detección basada en LOF (Local Outlier Factor)
            if len(volumes) > self.lookback_period:
                # Aplicar LOF
                lof = LocalOutlierFactor(n_neighbors=min(20, len(volumes) - 1))
                
                # Calcular factores de anomalía (-1 = anomalía, 1 = normal)
                lof_scores = lof.fit_predict(window_data)
                
                # Obtener puntuaciones de outlier (negativo = más anómalo)
                outlier_factors = lof.negative_outlier_factor_
                
                # Verificar último punto
                if lof_scores[-1] == -1:
                    # Normalizar score
                    normalized_score = min(1.0, abs(outlier_factors[-1]) / 2.0)
                    
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "volume_lof",
                        "score": normalized_score,
                        "details": {
                            "outlier_factor": float(outlier_factors[-1]),
                            "volume": volumes[-1],
                            "normalized_score": normalized_score
                        }
                    })
            
            # 3. Detección de picos súbitos
            if len(volumes) > 10:
                # Calcular media móvil
                window = 5
                moving_avg = np.convolve(volumes, np.ones(window)/window, mode='valid')
                
                # Último volumen vs. media móvil
                if len(moving_avg) > 0:
                    last_volume = volumes[-1]
                    last_ma = moving_avg[-1]
                    
                    if last_ma > 0 and last_volume > 3 * last_ma:
                        # Pico significativo (> 3x la media móvil)
                        anomalies.append({
                            "timestamp": timestamps[-1],
                            "type": "volume_spike",
                            "score": min(1.0, last_volume / (5 * last_ma)),
                            "details": {
                                "volume": last_volume,
                                "moving_avg": last_ma,
                                "ratio": last_volume / last_ma
                            }
                        })
        
        except Exception as e:
            self.logger.error(f"Error en detección de anomalías de volumen: {e}")
        
        return anomalies
    
    async def _detect_pattern_anomalies(
        self, 
        timestamps: List[float], 
        prices: List[float], 
        volumes: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Detectar patrones anómalos combinando precio y volumen.
        
        Args:
            timestamps: Lista de timestamps
            prices: Lista de precios
            volumes: Lista de volúmenes
            
        Returns:
            Lista de anomalías detectadas
        """
        anomalies = []
        
        try:
            # Necesitamos al menos algunos datos
            if len(prices) < 10 or len(volumes) < 10:
                return []
            
            # 1. Detección de divergencia precio-volumen
            # Calcular correlaciones de corto plazo (últimos 10 puntos)
            short_term = 10
            if len(prices) >= short_term:
                recent_prices = prices[-short_term:]
                recent_volumes = volumes[-short_term:]
                
                # Normalizar para correlación
                norm_prices = (np.array(recent_prices) - np.mean(recent_prices)) / np.std(recent_prices) if np.std(recent_prices) > 0 else np.zeros_like(recent_prices)
                norm_volumes = (np.array(recent_volumes) - np.mean(recent_volumes)) / np.std(recent_volumes) if np.std(recent_volumes) > 0 else np.zeros_like(recent_volumes)
                
                # Calcular correlación
                correlation = np.corrcoef(norm_prices, norm_volumes)[0, 1] if len(norm_prices) == len(norm_volumes) else 0
                
                # Divergencia fuerte (correlación negativa significativa)
                if correlation < -0.7:
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "price_volume_divergence",
                        "score": min(1.0, abs(correlation)),
                        "details": {
                            "correlation": correlation,
                            "price_change": (prices[-1] / prices[-short_term] - 1) if prices[-short_term] > 0 else 0,
                            "volume_change": (volumes[-1] / volumes[-short_term] - 1) if volumes[-short_term] > 0 else 0
                        }
                    })
            
            # 2. Detección de acumulación/distribución
            # Buscar volumen creciente con precio estable
            if len(prices) >= 20:
                # Calcular cambios porcentuales
                price_changes = np.array([prices[i] / prices[i-1] - 1 for i in range(1, len(prices))])
                volume_changes = np.array([volumes[i] / volumes[i-1] - 1 for i in range(1, len(volumes))])
                
                # Promedios recientes
                recent_price_volatility = np.std(price_changes[-10:])
                recent_volume_change = np.mean(volume_changes[-10:])
                
                # Acumulación: volumen creciente con precio estable
                if recent_price_volatility < 0.005 and recent_volume_change > 0.3:
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "accumulation_pattern",
                        "score": min(1.0, recent_volume_change / 0.5),
                        "details": {
                            "price_volatility": recent_price_volatility,
                            "volume_change": recent_volume_change
                        }
                    })
                
                # Distribución: volumen creciente con precios a la baja
                if np.mean(price_changes[-10:]) < -0.005 and recent_volume_change > 0.3:
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "distribution_pattern",
                        "score": min(1.0, recent_volume_change / 0.5),
                        "details": {
                            "price_change": np.mean(price_changes[-10:]),
                            "volume_change": recent_volume_change
                        }
                    })
            
            # 3. Detección de patrones de liquidez
            # Calcular ratio volumen/precio
            liquidity_ratio = [v / p if p > 0 else 0 for v, p in zip(volumes, prices)]
            
            if len(liquidity_ratio) > 10:
                # Cambios en ratio de liquidez
                avg_liquidity = np.mean(liquidity_ratio[:-1])
                last_liquidity = liquidity_ratio[-1]
                
                if avg_liquidity > 0 and last_liquidity > 3 * avg_liquidity:
                    # Anomalía de liquidez (aumento súbito)
                    anomalies.append({
                        "timestamp": timestamps[-1],
                        "type": "liquidity_change",
                        "score": min(1.0, last_liquidity / (5 * avg_liquidity)),
                        "details": {
                            "liquidity_ratio": last_liquidity,
                            "avg_liquidity": avg_liquidity,
                            "ratio": last_liquidity / avg_liquidity if avg_liquidity > 0 else 0
                        }
                    })
        
        except Exception as e:
            self.logger.error(f"Error en detección de patrones anómalos: {e}")
        
        return anomalies
    
    async def _generate_anomaly_chart(
        self,
        symbol: str,
        timestamps: List[float],
        prices: List[float],
        volumes: List[float],
        anomalies: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Generar gráfico para visualizar anomalías detectadas.
        
        Args:
            symbol: Símbolo analizado
            timestamps: Lista de timestamps
            prices: Lista de precios
            volumes: Lista de volúmenes
            anomalies: Lista de anomalías detectadas
            
        Returns:
            Ruta al archivo del gráfico, o None si falla
        """
        try:
            # Convertir timestamps a fechas para el gráfico
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Crear figura con subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Gráfico de precios
            ax1.plot(dates, prices, 'b-', linewidth=1.5, label='Precio')
            ax1.set_title(f'Anomalías Detectadas - {symbol}')
            ax1.set_ylabel('Precio')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de volumen
            ax2.bar(dates, volumes, width=0.8/(len(dates)), color='gray', alpha=0.5, label='Volumen')
            ax2.set_ylabel('Volumen')
            ax2.grid(True, alpha=0.3)
            
            # Marcar anomalías en el gráfico
            for anomaly in anomalies:
                ts = anomaly.get("timestamp")
                anomaly_type = anomaly.get("type", "")
                score = anomaly.get("score", 0)
                
                # Convertir timestamp a fecha
                date = datetime.fromtimestamp(ts)
                
                # Determinar el color y marcador según tipo
                if "price" in anomaly_type:
                    # Anomalía de precio (círculo rojo)
                    ax1.plot(date, prices[timestamps.index(ts)], 'ro', markersize=8+score*4, alpha=0.7)
                elif "volume" in anomaly_type:
                    # Anomalía de volumen (triángulo amarillo)
                    ax2.plot(date, volumes[timestamps.index(ts)], 'y^', markersize=8+score*4, alpha=0.7)
                else:
                    # Otro tipo de anomalía (diamante verde)
                    idx = timestamps.index(ts)
                    ax1.plot(date, prices[idx], 'gD', markersize=8+score*4, alpha=0.7)
                    ax2.plot(date, volumes[idx], 'gD', markersize=8+score*4, alpha=0.7)
            
            # Añadir leyenda
            ax1.legend()
            ax2.legend()
            
            # Configurar formato de fecha en el eje x
            plt.gcf().autofmt_xdate()
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar gráfico
            timestamp = int(time.time())
            file_path = f"{self.plots_dir}/{symbol}_anomalies_{timestamp}.png"
            plt.savefig(file_path)
            plt.close()
            
            return file_path
        
        except Exception as e:
            self.logger.error(f"Error al generar gráfico de anomalías: {e}")
            return None
    
    async def get_recent_anomalies(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener anomalías recientes para un símbolo.
        
        Args:
            symbol: Símbolo a consultar
            limit: Número máximo de resultados
            
        Returns:
            Lista de anomalías recientes
        """
        if symbol not in self.anomalies:
            return []
        
        # Ordenar por timestamp (más recientes primero)
        sorted_anomalies = sorted(self.anomalies[symbol], key=lambda x: x[0], reverse=True)
        
        # Limitar resultados
        recent = sorted_anomalies[:limit]
        
        # Convertir a formato de salida
        result = []
        for ts, a_type, score, details in recent:
            result.append({
                "timestamp": ts,
                "datetime": datetime.fromtimestamp(ts).isoformat(),
                "type": a_type,
                "score": score,
                "details": details
            })
        
        return result
    
    async def get_anomaly_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de anomalías para un símbolo.
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Estadísticas de anomalías
        """
        if symbol not in self.anomalies or not self.anomalies[symbol]:
            return {
                "symbol": symbol,
                "total_anomalies": 0,
                "types": {},
                "recent_activity": 0
            }
        
        # Contar anomalías por tipo
        type_counts = {}
        for _, a_type, _, _ in self.anomalies[symbol]:
            if a_type not in type_counts:
                type_counts[a_type] = 0
            type_counts[a_type] += 1
        
        # Calcular actividad reciente (últimas 24 horas)
        now = time.time()
        day_ago = now - 86400  # 24 horas en segundos
        
        recent_count = sum(1 for ts, _, _, _ in self.anomalies[symbol] if ts >= day_ago)
        
        # Obtener último timestamp
        last_anomaly = max(ts for ts, _, _, _ in self.anomalies[symbol])
        last_datetime = datetime.fromtimestamp(last_anomaly).isoformat()
        
        return {
            "symbol": symbol,
            "total_anomalies": len(self.anomalies[symbol]),
            "types": type_counts,
            "recent_activity": recent_count,
            "last_anomaly": last_datetime
        }
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Analizar un símbolo en busca de anomalías y generar reporte.
        
        Args:
            symbol: Símbolo a analizar
            
        Returns:
            Reporte de análisis
        """
        # Detectar anomalías
        anomalies = await self.detect_anomalies(symbol)
        
        # Obtener estadísticas
        stats = await self.get_anomaly_stats(symbol)
        
        # Obtener anomalías recientes
        recent = await self.get_recent_anomalies(symbol, 5)
        
        # Crear reporte
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "recent_anomalies": recent,
            "current_anomalies": anomalies,
            "action_recommendations": []
        }
        
        # Generar recomendaciones
        for anomaly in anomalies:
            # Solo recomendar para anomalías con score alto
            if anomaly.get("score", 0) > 0.6:
                anomaly_type = anomaly.get("type", "")
                
                recommendation = {
                    "anomaly_type": anomaly_type,
                    "confidence": anomaly.get("score", 0),
                }
                
                # Específicas según tipo
                if "price_gap" in anomaly_type:
                    direction = anomaly.get("details", {}).get("direction", "")
                    recommendation["action"] = f"Posible {'compra' if direction == 'up' else 'venta'} por gap de precio"
                    recommendation["rationale"] = f"Detectado gap significativo de precio hacia {direction}"
                
                elif "volume_spike" in anomaly_type:
                    recommendation["action"] = "Monitorear para confirmación"
                    recommendation["rationale"] = "Incremento anormal de volumen sin cambio significativo de precio"
                
                elif "accumulation_pattern" in anomaly_type:
                    recommendation["action"] = "Considerar compra"
                    recommendation["rationale"] = "Patrón de acumulación detectado: volumen creciente con precio estable"
                
                elif "distribution_pattern" in anomaly_type:
                    recommendation["action"] = "Considerar venta"
                    recommendation["rationale"] = "Patrón de distribución detectado: volumen creciente con precio a la baja"
                
                elif "price_volume_divergence" in anomaly_type:
                    recommendation["action"] = "Prepararse para reversión"
                    recommendation["rationale"] = "Divergencia significativa entre precio y volumen"
                
                else:
                    # Genérica para otros tipos
                    recommendation["action"] = "Monitorear condiciones anormales"
                    recommendation["rationale"] = f"Anomalía detectada: {anomaly_type}"
                
                report["action_recommendations"].append(recommendation)
        
        return report


# Exportación para uso fácil
anomaly_detector = AnomalyDetector()