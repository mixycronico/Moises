"""
Detector de anomalías para el sistema Genesis.

Este módulo implementa algoritmos para detectar comportamientos anómalos
en los datos de mercado, que podrían indicar manipulación, condiciones
inusuales o oportunidades de trading.
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

from sklearn.ensemble import IsolationForest
from genesis.core.base import Component

class AnomalyDetector(Component):
    """
    Detector de anomalías para series temporales de precios y volumen.
    
    Esta clase implementa varios algoritmos para identificar comportamientos
    anómalos en los datos de mercado, como movimientos de precio inusuales,
    picos de volumen, y patrones de actividad sospechosos.
    """
    
    def __init__(self, name: str = "anomaly_detector"):
        """
        Inicializar el detector de anomalías.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self.anomaly_thresholds = {
            "price_volatility": 2.5,    # Desviaciones estándar
            "volume_spike": 3.0,        # Desviaciones estándar
            "price_gap": 0.02,          # 2% o más
            "low_liquidity": 0.5        # 50% menos liquidez que la media
        }
        self.models = {}
        self.historical_anomalies = {}
        
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
            source: Componente origen
        """
        if event_type == "market_data.update":
            # Procesar actualización de datos de mercado
            symbol = data.get("symbol")
            market_data = data.get("data")
            
            if symbol and market_data:
                # Analizar para detectar anomalías
                anomalies = await self.detect_anomalies(symbol, market_data)
                
                if anomalies:
                    # Emitir evento de anomalía detectada
                    await self.emit_event("anomaly.detected", {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "anomalies": anomalies
                    })
                    
        elif event_type == "config.update":
            # Actualizar umbrales de detección
            thresholds = data.get("anomaly_thresholds")
            if thresholds:
                self.anomaly_thresholds.update(thresholds)
                self.logger.info(f"Umbrales de anomalías actualizados: {self.anomaly_thresholds}")
                
    async def detect_anomalies(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detectar anomalías en los datos de mercado.
        
        Args:
            symbol: Símbolo de trading
            market_data: Datos de mercado recientes
            
        Returns:
            Diccionario con anomalías detectadas
        """
        anomalies = {}
        
        # Detectar diferentes tipos de anomalías
        price_anomalies = await self.detect_price_anomalies(symbol, market_data)
        volume_anomalies = await self.detect_volume_anomalies(symbol, market_data)
        pattern_anomalies = await self.detect_pattern_anomalies(symbol, market_data)
        liquidity_anomalies = await self.detect_liquidity_anomalies(symbol, market_data)
        
        # Combinar resultados
        anomalies.update(price_anomalies)
        anomalies.update(volume_anomalies)
        anomalies.update(pattern_anomalies)
        anomalies.update(liquidity_anomalies)
        
        # Registrar anomalías detectadas
        if anomalies:
            if symbol not in self.historical_anomalies:
                self.historical_anomalies[symbol] = []
                
            self.historical_anomalies[symbol].append({
                "timestamp": datetime.now().isoformat(),
                "anomalies": anomalies
            })
            
            # Mantener solo las últimas 100 anomalías por símbolo
            if len(self.historical_anomalies[symbol]) > 100:
                self.historical_anomalies[symbol] = self.historical_anomalies[symbol][-100:]
                
            self.logger.info(f"Anomalías detectadas para {symbol}: {anomalies}")
            
        return anomalies
        
    async def detect_price_anomalies(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detectar anomalías en el precio.
        
        Args:
            symbol: Símbolo de trading
            market_data: Datos de mercado recientes
            
        Returns:
            Anomalías de precio detectadas
        """
        result = {}
        
        # Verificar si tenemos los datos necesarios
        if not market_data.get("price_data"):
            return result
            
        price_data = market_data["price_data"]
        
        # Verificar datos mínimos
        if len(price_data) < 10:
            return result
            
        # Datos recientes para análisis
        recent_prices = [p.get("close", p.get("price", 0)) for p in price_data[-20:]]
        
        if not recent_prices:
            return result
            
        # Calcular rendimientos
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                returns.append((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1])
            else:
                returns.append(0)
                
        # Detectar saltos de precio (gaps)
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                price_change = abs((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1])
                if price_change > self.anomaly_thresholds["price_gap"]:
                    result["price_gap"] = {
                        "prev_price": recent_prices[i-1],
                        "current_price": recent_prices[i],
                        "change_pct": price_change * 100,
                        "direction": "up" if recent_prices[i] > recent_prices[i-1] else "down"
                    }
                    break
                    
        # Detectar volatilidad inusual
        if len(returns) > 5:
            volatility = np.std(returns) * np.sqrt(252)  # Anualizada
            avg_volatility = market_data.get("avg_volatility", volatility / 2)
            
            if volatility > avg_volatility * self.anomaly_thresholds["price_volatility"]:
                result["high_volatility"] = {
                    "current_volatility": volatility,
                    "avg_volatility": avg_volatility,
                    "ratio": volatility / avg_volatility if avg_volatility > 0 else float('inf')
                }
                
        # Detectar precios que se desvían significativamente de la media móvil
        if len(recent_prices) > 10:
            sma = sum(recent_prices[-10:]) / 10
            latest_price = recent_prices[-1]
            
            if sma > 0:
                deviation = abs((latest_price - sma) / sma)
                if deviation > 0.05:  # 5% o más
                    result["price_deviation"] = {
                        "price": latest_price,
                        "sma": sma,
                        "deviation_pct": deviation * 100,
                        "direction": "up" if latest_price > sma else "down"
                    }
                    
        return result
        
    async def detect_volume_anomalies(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detectar anomalías en el volumen.
        
        Args:
            symbol: Símbolo de trading
            market_data: Datos de mercado recientes
            
        Returns:
            Anomalías de volumen detectadas
        """
        result = {}
        
        # Verificar si tenemos los datos necesarios
        if not market_data.get("price_data"):
            return result
            
        price_data = market_data["price_data"]
        
        # Verificar datos mínimos
        if len(price_data) < 10:
            return result
            
        # Extraer datos de volumen
        volumes = [p.get("volume", 0) for p in price_data[-30:]]
        
        if not volumes or all(v == 0 for v in volumes):
            return result
            
        # Calcular estadísticas
        avg_volume = sum(volumes[:-1]) / (len(volumes) - 1) if len(volumes) > 1 else volumes[0]
        std_volume = np.std(volumes[:-1]) if len(volumes) > 2 else avg_volume * 0.1
        latest_volume = volumes[-1]
        
        # Detectar picos de volumen
        if std_volume > 0:
            z_score = (latest_volume - avg_volume) / std_volume
            if z_score > self.anomaly_thresholds["volume_spike"]:
                result["volume_spike"] = {
                    "current_volume": latest_volume,
                    "avg_volume": avg_volume,
                    "ratio": latest_volume / avg_volume if avg_volume > 0 else float('inf'),
                    "z_score": z_score
                }
                
        # Detectar caídas drásticas de volumen
        if avg_volume > 0 and latest_volume < avg_volume * 0.2:  # 80% menos que la media
            result["low_volume"] = {
                "current_volume": latest_volume,
                "avg_volume": avg_volume,
                "ratio": latest_volume / avg_volume
            }
            
        # Detectar patrones inusuales en la secuencia de volumen
        if len(volumes) > 15:
            # Usar modelo de isolation forest si ya existe
            if f"{symbol}_volume" in self.models:
                model = self.models[f"{symbol}_volume"]
                # Predecir anomalías para los últimos 5 valores
                recent_volumes = np.array(volumes[-5:]).reshape(-1, 1)
                predictions = model.predict(recent_volumes)
                
                if -1 in predictions:  # -1 indica anomalía
                    result["volume_pattern_anomaly"] = {
                        "recent_volumes": volumes[-5:],
                        "avg_volume": avg_volume
                    }
            else:
                # Crear y entrenar un nuevo modelo
                try:
                    model = IsolationForest(contamination=0.05, random_state=42)
                    model.fit(np.array(volumes).reshape(-1, 1))
                    self.models[f"{symbol}_volume"] = model
                except Exception as e:
                    self.logger.error(f"Error al entrenar modelo para {symbol}: {e}")
                    
        return result
        
    async def detect_pattern_anomalies(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detectar patrones anómalos en los datos.
        
        Args:
            symbol: Símbolo de trading
            market_data: Datos de mercado recientes
            
        Returns:
            Anomalías de patrones detectadas
        """
        result = {}
        
        # Verificar si tenemos los datos necesarios
        if not market_data.get("price_data"):
            return result
            
        price_data = market_data["price_data"]
        
        # Verificar datos mínimos
        if len(price_data) < 20:
            return result
            
        # Extraer datos
        closes = [p.get("close", p.get("price", 0)) for p in price_data[-20:]]
        highs = [p.get("high", p.get("close", 0)) for p in price_data[-20:]]
        lows = [p.get("low", p.get("close", 0)) for p in price_data[-20:]]
        
        # Detectar patrones de vela específicos
        # Ejemplo: Doji (cuerpo muy pequeño)
        for i in range(max(0, len(closes) - 3), len(closes)):
            if i > 0:
                open_price = price_data[i].get("open", closes[i-1])
                close_price = closes[i]
                high_price = highs[i]
                low_price = lows[i]
                
                body_size = abs(close_price - open_price)
                wick_size = high_price - max(open_price, close_price) + min(open_price, close_price) - low_price
                
                # Detectar Doji
                if high_price > low_price and body_size < (high_price - low_price) * 0.1:
                    result["doji_pattern"] = {
                        "position": i,
                        "open": open_price,
                        "close": close_price,
                        "high": high_price,
                        "low": low_price
                    }
                    break
                    
                # Detectar Hammer
                if (body_size > 0 and 
                    min(open_price, close_price) - low_price > body_size * 2 and
                    high_price - max(open_price, close_price) < body_size * 0.2):
                    result["hammer_pattern"] = {
                        "position": i,
                        "open": open_price,
                        "close": close_price,
                        "high": high_price,
                        "low": low_price
                    }
                    break
                    
        # Detectar divergencias
        if len(closes) >= 20 and "indicators" in market_data:
            indicators = market_data["indicators"]
            
            # Divergencia RSI
            if "rsi" in indicators and len(indicators["rsi"]) >= 20:
                rsi_values = indicators["rsi"][-20:]
                
                # Buscar divergencia alcista
                price_lows = []
                rsi_lows = []
                
                for i in range(1, len(closes) - 1):
                    # Mínimo local en precio
                    if closes[i] < closes[i-1] and closes[i] < closes[i+1]:
                        price_lows.append((i, closes[i]))
                        
                    # Mínimo local en RSI
                    if i < len(rsi_values) and i > 0 and i < len(rsi_values) - 1:
                        if rsi_values[i] < rsi_values[i-1] and rsi_values[i] < rsi_values[i+1]:
                            rsi_lows.append((i, rsi_values[i]))
                            
                # Verificar divergencia
                if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                    last_price_lows = sorted(price_lows[-2:], key=lambda x: x[0])
                    last_rsi_lows = sorted(rsi_lows[-2:], key=lambda x: x[0])
                    
                    if (len(last_price_lows) == 2 and len(last_rsi_lows) == 2 and
                        last_price_lows[1][1] < last_price_lows[0][1] and  # Precio: lower low
                        last_rsi_lows[1][1] > last_rsi_lows[0][1]):        # RSI: higher low
                        result["bullish_divergence"] = {
                            "price_lows": last_price_lows,
                            "rsi_lows": last_rsi_lows
                        }
                        
        return result
        
    async def detect_liquidity_anomalies(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detectar anomalías en la liquidez.
        
        Args:
            symbol: Símbolo de trading
            market_data: Datos de mercado recientes
            
        Returns:
            Anomalías de liquidez detectadas
        """
        result = {}
        
        # Verificar si tenemos datos de libro de órdenes
        if not market_data.get("orderbook"):
            return result
            
        orderbook = market_data["orderbook"]
        
        # Verificar campos necesarios
        if "bids" not in orderbook or "asks" not in orderbook:
            return result
            
        bids = orderbook["bids"]
        asks = orderbook["asks"]
        
        # Calcular profundidad de mercado
        bid_liquidity = sum(amount for price, amount in bids[:5]) if len(bids) >= 5 else 0
        ask_liquidity = sum(amount for price, amount in asks[:5]) if len(asks) >= 5 else 0
        
        # Calcular spread
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        
        if best_bid > 0 and best_ask > 0:
            spread = (best_ask - best_bid) / best_bid
            spread_pct = spread * 100
            
            # Obtener spread promedio si está disponible
            avg_spread = market_data.get("avg_spread", spread * 2)
            
            # Detectar spread anormalmente alto
            if spread > avg_spread * 2:
                result["high_spread"] = {
                    "current_spread": spread_pct,
                    "avg_spread": avg_spread * 100,
                    "ratio": spread / avg_spread
                }
                
        # Detectar liquidez anormalmente baja
        avg_bid_liquidity = market_data.get("avg_bid_liquidity", bid_liquidity * 2)
        avg_ask_liquidity = market_data.get("avg_ask_liquidity", ask_liquidity * 2)
        
        if avg_bid_liquidity > 0 and bid_liquidity < avg_bid_liquidity * self.anomaly_thresholds["low_liquidity"]:
            result["low_bid_liquidity"] = {
                "current_liquidity": bid_liquidity,
                "avg_liquidity": avg_bid_liquidity,
                "ratio": bid_liquidity / avg_bid_liquidity
            }
            
        if avg_ask_liquidity > 0 and ask_liquidity < avg_ask_liquidity * self.anomaly_thresholds["low_liquidity"]:
            result["low_ask_liquidity"] = {
                "current_liquidity": ask_liquidity,
                "avg_liquidity": avg_ask_liquidity,
                "ratio": ask_liquidity / avg_ask_liquidity
            }
            
        # Detectar desequilibrio severo entre compras y ventas
        if bid_liquidity > 0 and ask_liquidity > 0:
            ratio = bid_liquidity / ask_liquidity
            if ratio > 3 or ratio < 0.33:
                result["liquidity_imbalance"] = {
                    "bid_liquidity": bid_liquidity,
                    "ask_liquidity": ask_liquidity,
                    "ratio": ratio,
                    "bias": "buy" if ratio > 1 else "sell"
                }
                
        return result
        
    async def train_models(self, symbol: str, historical_data: Dict[str, Any]) -> None:
        """
        Entrenar modelos de detección de anomalías con datos históricos.
        
        Args:
            symbol: Símbolo de trading
            historical_data: Datos históricos para entrenamiento
        """
        try:
            # Entrenar para volumen
            if "price_data" in historical_data and len(historical_data["price_data"]) > 50:
                volumes = [p.get("volume", 0) for p in historical_data["price_data"] if p.get("volume", 0) > 0]
                
                if volumes:
                    model = IsolationForest(contamination=0.05, random_state=42)
                    model.fit(np.array(volumes).reshape(-1, 1))
                    self.models[f"{symbol}_volume"] = model
                    self.logger.info(f"Modelo de volumen entrenado para {symbol}")
                    
            # Entrenar para precios
            if "price_data" in historical_data and len(historical_data["price_data"]) > 50:
                closes = [p.get("close", p.get("price", 0)) for p in historical_data["price_data"]]
                
                if closes and all(c > 0 for c in closes):
                    # Calcular rendimientos
                    returns = []
                    for i in range(1, len(closes)):
                        returns.append((closes[i] - closes[i-1]) / closes[i-1])
                        
                    if returns:
                        model = IsolationForest(contamination=0.05, random_state=42)
                        model.fit(np.array(returns).reshape(-1, 1))
                        self.models[f"{symbol}_returns"] = model
                        self.logger.info(f"Modelo de rendimientos entrenado para {symbol}")
                        
        except Exception as e:
            self.logger.error(f"Error al entrenar modelos para {symbol}: {e}")
            try:
                await self.emit_event("anomaly.training_error", {
                    "symbol": symbol,
                    "error": str(e)
                })
            except Exception as e:
                self.logger.error(f"Error al emitir evento de error: {e}")
        
    def get_historical_anomalies(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener anomalías históricas para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            limit: Número máximo de anomalías a retornar
            
        Returns:
            Lista de anomalías históricas
        """
        if symbol not in self.historical_anomalies:
            return []
            
        return self.historical_anomalies[symbol][-limit:]
        
    def update_threshold(self, anomaly_type: str, new_value: float) -> bool:
        """
        Actualizar umbral para un tipo de anomalía.
        
        Args:
            anomaly_type: Tipo de anomalía
            new_value: Nuevo valor de umbral
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        if anomaly_type in self.anomaly_thresholds:
            self.anomaly_thresholds[anomaly_type] = new_value
            self.logger.info(f"Umbral para {anomaly_type} actualizado a {new_value}")
            return True
        return False