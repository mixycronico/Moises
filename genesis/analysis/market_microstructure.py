"""
Análisis de microestructura de mercado para el sistema Genesis.

Este módulo proporciona herramientas para analizar la microestructura del
mercado, incluyendo impacto de precio, estimación de liquidez, y 
detección de patrones de trading de alta frecuencia.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import asyncio
import time
from datetime import datetime, timedelta
import os

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class MarketMicrostructureAnalyzer(Component):
    """
    Analizador de microestructura de mercado.
    
    Este componente analiza la microestructura del mercado, incluyendo
    liquidez, impacto de precio, y patrones de trading de alta frecuencia.
    """
    
    def __init__(self, name: str = "market_microstructure"):
        """
        Inicializar el analizador de microestructura.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Configuración
        self.window_size = 50  # Ventana de datos para el análisis
        self.tick_threshold = 0.0001  # Umbral para considerar un cambio de tick
        self.volume_threshold = 0.1  # Umbral para volumen significativo (10% del volumen medio)
        
        # Datos de market
        # Symbol -> (timestamp, price, volume)
        self.price_data: Dict[str, List[Tuple[float, float, float]]] = {}
        
        # Datos de trades
        # Symbol -> (timestamp, trade_data)
        self.trades: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        
        # Datos de OrderBook
        # Symbol -> (timestamp, book_data)
        self.order_books: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        
        # Resultados de análisis
        # Symbol -> análisis
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        
        # Directorio para gráficos
        self.plot_dir = "data/plots/microstructure"
        os.makedirs(self.plot_dir, exist_ok=True)
    
    async def start(self) -> None:
        """Iniciar el analizador de microestructura."""
        await super().start()
        self.logger.info("Analizador de microestructura de mercado iniciado")
    
    async def stop(self) -> None:
        """Detener el analizador de microestructura."""
        await super().stop()
        self.logger.info("Analizador de microestructura de mercado detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Procesar actualizaciones de precio
        if event_type == "market.ticker_updated":
            await self._process_price_update(data)
        
        # Procesar datos de trades
        elif event_type == "market.trade_executed":
            await self._process_trade(data)
        
        # Procesar datos del OrderBook
        elif event_type == "market.orderbook_updated":
            await self._process_orderbook(data)
    
    async def _process_price_update(self, data: Dict[str, Any]) -> None:
        """
        Procesar actualización de precio.
        
        Args:
            data: Datos de la actualización
        """
        symbol = data.get("symbol")
        if not symbol:
            return
        
        timestamp = data.get("timestamp", time.time())
        price = data.get("price", 0.0)
        volume = data.get("volume", 0.0)
        
        # Inicializar lista si es necesario
        if symbol not in self.price_data:
            self.price_data[symbol] = []
        
        # Añadir datos
        self.price_data[symbol].append((timestamp, price, volume))
        
        # Mantener solo los últimos N datos
        self.price_data[symbol] = self.price_data[symbol][-self.window_size:]
        
        # Analizar si tenemos suficientes datos
        if len(self.price_data[symbol]) >= 10:  # Mínimo 10 muestras para análisis
            await self._analyze_price_action(symbol)
    
    async def _process_trade(self, data: Dict[str, Any]) -> None:
        """
        Procesar trade ejecutado.
        
        Args:
            data: Datos del trade
        """
        symbol = data.get("symbol")
        if not symbol:
            return
        
        timestamp = data.get("timestamp", time.time())
        
        # Inicializar lista si es necesario
        if symbol not in self.trades:
            self.trades[symbol] = []
        
        # Añadir datos
        self.trades[symbol].append((timestamp, data))
        
        # Mantener solo los últimos N datos
        self.trades[symbol] = self.trades[symbol][-self.window_size:]
        
        # Analizar si tenemos suficientes datos
        if len(self.trades[symbol]) >= 10:  # Mínimo 10 muestras para análisis
            await self._analyze_trades_microstructure(symbol)
    
    async def _process_orderbook(self, data: Dict[str, Any]) -> None:
        """
        Procesar actualización de OrderBook.
        
        Args:
            data: Datos del OrderBook
        """
        symbol = data.get("symbol")
        if not symbol:
            return
        
        timestamp = data.get("timestamp", time.time())
        
        # Inicializar lista si es necesario
        if symbol not in self.order_books:
            self.order_books[symbol] = []
        
        # Añadir datos
        self.order_books[symbol].append((timestamp, data))
        
        # Mantener solo los últimos N datos
        self.order_books[symbol] = self.order_books[symbol][-self.window_size:]
        
        # Analizar si tenemos suficientes datos
        if len(self.order_books[symbol]) >= 10:  # Mínimo 10 muestras para análisis
            await self._analyze_orderbook_microstructure(symbol)
    
    async def _analyze_price_action(self, symbol: str) -> None:
        """
        Analizar acción del precio.
        
        Args:
            symbol: Símbolo a analizar
        """
        # Verificar datos
        if not self.price_data.get(symbol):
            return
        
        # Inicializar resultados para este símbolo si es necesario
        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        
        # Extraer datos de precio y volumen
        timestamps = [d[0] for d in self.price_data[symbol]]
        prices = [d[1] for d in self.price_data[symbol]]
        volumes = [d[2] for d in self.price_data[symbol]]
        
        # Convertir a arrays de numpy para cálculos más eficientes
        prices_array = np.array(prices)
        volumes_array = np.array(volumes)
        
        # Calcular retornos
        returns = np.diff(prices_array) / prices_array[:-1]
        
        # Calcular volatilidad (desviación estándar de retornos)
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Calcular métricas de microestructura
        
        # 1. Ratio de volatilidad por tick
        tick_volatility = volatility / np.mean(np.abs(np.diff(prices_array))) if len(prices_array) > 1 else 0
        
        # 2. Autocorrelación de retornos
        autocorr_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        
        # 3. Autocorrelación del volumen
        volume_changes = np.diff(volumes_array)
        volume_autocorr = np.corrcoef(volume_changes[:-1], volume_changes[1:])[0, 1] if len(volume_changes) > 1 else 0
        
        # 4. Detección de patrones de mean-reversion vs momentum
        is_mean_reverting = autocorr_1 < -0.2  # Umbral negativo indica mean-reversion
        is_momentum = autocorr_1 > 0.2  # Umbral positivo indica momentum
        
        # 5. Clustering de volatilidad
        returns_squared = returns ** 2
        volatility_clustering = np.corrcoef(returns_squared[:-1], returns_squared[1:])[0, 1] if len(returns_squared) > 1 else 0
        
        # 6. Relación volumen-volatilidad
        vol_vol_corr = np.corrcoef(np.abs(returns), volumes_array[1:])[0, 1] if len(returns) > 0 else 0
        
        # Guardar resultados
        result = {
            "timestamp": datetime.now().isoformat(),
            "volatility": volatility,
            "tick_volatility": tick_volatility,
            "autocorrelation": autocorr_1,
            "volume_autocorrelation": volume_autocorr,
            "is_mean_reverting": is_mean_reverting,
            "is_momentum": is_momentum,
            "volatility_clustering": volatility_clustering,
            "volume_volatility_correlation": vol_vol_corr
        }
        
        # Actualizar resultados
        self.analysis_results[symbol]["price_action"] = result
        
        # Emitir eventos basados en el análisis
        
        # Evento de régimen de mercado
        if is_mean_reverting:
            await self.emit_event("analysis.market_regime", {
                "symbol": symbol,
                "regime": "mean_reversion",
                "confidence": abs(autocorr_1),
                "timestamp": datetime.now().isoformat()
            })
            self.logger.info(f"Régimen de mean-reversion detectado en {symbol} (autocorr: {autocorr_1:.2f})")
        elif is_momentum:
            await self.emit_event("analysis.market_regime", {
                "symbol": symbol,
                "regime": "momentum",
                "confidence": autocorr_1,
                "timestamp": datetime.now().isoformat()
            })
            self.logger.info(f"Régimen de momentum detectado en {symbol} (autocorr: {autocorr_1:.2f})")
        
        # Evento de clustering de volatilidad
        if volatility_clustering > 0.3:  # Umbral significativo
            await self.emit_event("analysis.volatility_clustering", {
                "symbol": symbol,
                "clustering": volatility_clustering,
                "volatility": volatility,
                "timestamp": datetime.now().isoformat()
            })
            self.logger.info(f"Clustering de volatilidad detectado en {symbol} ({volatility_clustering:.2f})")
    
    async def _analyze_trades_microstructure(self, symbol: str) -> None:
        """
        Analizar microestructura de trades.
        
        Args:
            symbol: Símbolo a analizar
        """
        # Verificar datos
        if not self.trades.get(symbol):
            return
        
        # Inicializar resultados para este símbolo si es necesario
        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        
        # Extraer datos de trades
        trades = [t[1] for t in self.trades[symbol]]
        
        # Calcular métricas
        
        # 1. Frecuencia de trades
        timestamps = [t.get("timestamp", 0) for t in trades]
        if len(timestamps) > 1:
            durations = np.diff(timestamps)
            avg_trade_interval = np.mean(durations)
            trade_frequency = 1 / avg_trade_interval if avg_trade_interval > 0 else 0
        else:
            avg_trade_interval = 0
            trade_frequency = 0
        
        # 2. Tamaño medio de trades
        sizes = [t.get("amount", 0) for t in trades]
        avg_trade_size = np.mean(sizes) if sizes else 0
        
        # 3. Distribución de tamaños
        size_std = np.std(sizes) if len(sizes) > 1 else 0
        
        # 4. Proporción de trades agresores
        # Un trade agresor toma liquidez (market order)
        aggressive_trades = [t for t in trades if t.get("taker_side") == t.get("side")]
        passive_trades = [t for t in trades if t.get("taker_side") != t.get("side")]
        
        aggressive_ratio = len(aggressive_trades) / len(trades) if trades else 0
        
        # 5. Detección de patrones de iceberg
        # Iceberg: series de trades pequeños al mismo precio en poco tiempo
        iceberg_threshold = 3  # Mínimo número de trades para considerar un iceberg
        iceberg_time_window = 5  # Ventana de tiempo en segundos
        
        # Agrupar trades por precio y proximidad temporal
        iceberg_candidates = []
        current_price = None
        current_trades = []
        
        for i, trade in enumerate(trades):
            if i == 0:
                current_price = trade.get("price", 0)
                current_trades = [trade]
                continue
            
            # Si mismo precio y dentro de la ventana temporal
            if (trade.get("price", 0) == current_price and 
                trade.get("timestamp", 0) - trades[i-1].get("timestamp", 0) < iceberg_time_window):
                current_trades.append(trade)
            else:
                # Evaluar grupo anterior
                if len(current_trades) >= iceberg_threshold:
                    iceberg_candidates.append(current_trades)
                
                # Iniciar nuevo grupo
                current_price = trade.get("price", 0)
                current_trades = [trade]
        
        # Evaluar último grupo
        if len(current_trades) >= iceberg_threshold:
            iceberg_candidates.append(current_trades)
        
        # 6. Detección de patrones de flash crashes/rallies
        price_changes = [trades[i].get("price", 0) - trades[i-1].get("price", 0) 
                         for i in range(1, len(trades))]
        
        # Un flash crash/rally es un movimiento extremo en poco tiempo
        if price_changes:
            max_price_change = max(abs(pc) for pc in price_changes)
            avg_price = np.mean([t.get("price", 0) for t in trades])
            flash_threshold = avg_price * 0.01  # 1% movimiento
            
            has_flash_pattern = max_price_change > flash_threshold
        else:
            has_flash_pattern = False
            max_price_change = 0
        
        # Guardar resultados
        result = {
            "timestamp": datetime.now().isoformat(),
            "trade_frequency": trade_frequency,
            "avg_trade_interval": avg_trade_interval,
            "avg_trade_size": avg_trade_size,
            "size_std": size_std,
            "aggressive_ratio": aggressive_ratio,
            "iceberg_patterns": len(iceberg_candidates),
            "has_flash_pattern": has_flash_pattern,
            "max_price_change": max_price_change
        }
        
        # Actualizar resultados
        self.analysis_results[symbol]["trades_microstructure"] = result
        
        # Emitir eventos basados en el análisis
        
        # Evento de detección de icebergs
        if iceberg_candidates:
            await self.emit_event("analysis.iceberg_detected", {
                "symbol": symbol,
                "count": len(iceberg_candidates),
                "details": [
                    {
                        "price": group[0].get("price", 0),
                        "side": group[0].get("side", ""),
                        "count": len(group),
                        "total_volume": sum(t.get("amount", 0) for t in group)
                    }
                    for group in iceberg_candidates
                ],
                "timestamp": datetime.now().isoformat()
            })
            self.logger.info(f"Patrón de iceberg detectado en {symbol}: {len(iceberg_candidates)} patrones")
        
        # Evento de flash pattern
        if has_flash_pattern:
            direction = "up" if max(price_changes, key=abs) > 0 else "down"
            await self.emit_event("analysis.flash_pattern", {
                "symbol": symbol,
                "direction": direction,
                "magnitude": max_price_change,
                "relative_magnitude": max_price_change / avg_price if avg_price > 0 else 0,
                "timestamp": datetime.now().isoformat()
            })
            self.logger.info(f"Patrón flash {direction} detectado en {symbol}: {max_price_change:.6f}")
    
    async def _analyze_orderbook_microstructure(self, symbol: str) -> None:
        """
        Analizar microestructura del OrderBook.
        
        Args:
            symbol: Símbolo a analizar
        """
        # Verificar datos
        if not self.order_books.get(symbol):
            return
        
        # Inicializar resultados para este símbolo si es necesario
        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        
        # Extraer datos de OrderBook
        order_books = [ob[1] for ob in self.order_books[symbol]]
        
        # Verificar que haya datos válidos
        if not order_books or not all(ob.get("bids") and ob.get("asks") for ob in order_books):
            return
        
        # Métricas de microestructura del OrderBook
        
        # 1. Spread promedio
        spreads = []
        for ob in order_books:
            if ob.get("bids") and ob.get("asks"):
                best_bid = ob["bids"][0][0] if ob["bids"] else 0
                best_ask = ob["asks"][0][0] if ob["asks"] else 0
                if best_bid > 0 and best_ask > 0:
                    spread = best_ask - best_bid
                    relative_spread = spread / best_bid
                    spreads.append(relative_spread)
        
        avg_spread = np.mean(spreads) if spreads else 0
        
        # 2. Profundidad del mercado
        # Volumen acumulado hasta cierta distancia del mid price
        depth_levels = 5  # Número de niveles a considerar
        
        bid_depths = []
        ask_depths = []
        
        for ob in order_books:
            if ob.get("bids") and ob.get("asks"):
                # Calcular mid price
                best_bid = ob["bids"][0][0] if ob["bids"] else 0
                best_ask = ob["asks"][0][0] if ob["asks"] else 0
                mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
                
                if mid_price > 0:
                    # Calcular profundidad
                    bid_depth = sum(bid[1] for bid in ob["bids"][:min(depth_levels, len(ob["bids"]))])
                    ask_depth = sum(ask[1] for ask in ob["asks"][:min(depth_levels, len(ob["asks"]))])
                    
                    bid_depths.append(bid_depth)
                    ask_depths.append(ask_depth)
        
        avg_bid_depth = np.mean(bid_depths) if bid_depths else 0
        avg_ask_depth = np.mean(ask_depths) if ask_depths else 0
        
        # 3. Asimetría de liquidez
        depth_ratio = avg_bid_depth / avg_ask_depth if avg_ask_depth > 0 else 0
        
        # 4. Estabilidad del OrderBook
        # Cambios en el nivel superior
        top_level_changes = []
        
        for i in range(1, len(order_books)):
            prev_ob = order_books[i-1]
            curr_ob = order_books[i]
            
            if prev_ob.get("bids") and curr_ob.get("bids") and prev_ob.get("asks") and curr_ob.get("asks"):
                prev_best_bid = prev_ob["bids"][0][0] if prev_ob["bids"] else 0
                curr_best_bid = curr_ob["bids"][0][0] if curr_ob["bids"] else 0
                
                prev_best_ask = prev_ob["asks"][0][0] if prev_ob["asks"] else 0
                curr_best_ask = curr_ob["asks"][0][0] if curr_ob["asks"] else 0
                
                if prev_best_bid > 0 and curr_best_bid > 0:
                    bid_change = (curr_best_bid - prev_best_bid) / prev_best_bid
                    top_level_changes.append(abs(bid_change))
                
                if prev_best_ask > 0 and curr_best_ask > 0:
                    ask_change = (curr_best_ask - prev_best_ask) / prev_best_ask
                    top_level_changes.append(abs(ask_change))
        
        top_level_volatility = np.mean(top_level_changes) if top_level_changes else 0
        
        # 5. Resistencia del OrderBook
        # Calculamos cuánto volumen se necesita para mover el precio un X%
        price_impact_threshold = 0.001  # 0.1%
        
        bid_impact_volumes = []
        ask_impact_volumes = []
        
        for ob in order_books:
            if ob.get("bids") and ob.get("asks"):
                best_bid = ob["bids"][0][0] if ob["bids"] else 0
                best_ask = ob["asks"][0][0] if ob["asks"] else 0
                
                if best_bid > 0 and best_ask > 0:
                    # Precio objetivo para impacto
                    bid_target = best_bid * (1 - price_impact_threshold)
                    ask_target = best_ask * (1 + price_impact_threshold)
                    
                    # Acumular volumen hasta alcanzar el precio objetivo
                    bid_volume = 0
                    for bid in ob["bids"]:
                        if bid[0] >= bid_target:
                            bid_volume += bid[1]
                        else:
                            break
                    
                    ask_volume = 0
                    for ask in ob["asks"]:
                        if ask[0] <= ask_target:
                            ask_volume += ask[1]
                        else:
                            break
                    
                    bid_impact_volumes.append(bid_volume)
                    ask_impact_volumes.append(ask_volume)
        
        avg_bid_impact = np.mean(bid_impact_volumes) if bid_impact_volumes else 0
        avg_ask_impact = np.mean(ask_impact_volumes) if ask_impact_volumes else 0
        
        # 6. Detección de paredes de liquidez
        liquidity_walls = []
        
        for ob in order_books:
            if ob.get("bids") and ob.get("asks"):
                # Obtener volúmenes medios
                bid_volumes = [bid[1] for bid in ob["bids"]]
                ask_volumes = [ask[1] for ask in ob["asks"]]
                
                avg_bid_vol = np.mean(bid_volumes) if bid_volumes else 0
                avg_ask_vol = np.mean(ask_volumes) if ask_volumes else 0
                
                # Detectar órdenes grandes (>3x el promedio)
                wall_threshold = 3
                
                bid_walls = [
                    {"price": ob["bids"][i][0], "volume": ob["bids"][i][1]}
                    for i in range(len(ob["bids"]))
                    if ob["bids"][i][1] > avg_bid_vol * wall_threshold
                ]
                
                ask_walls = [
                    {"price": ob["asks"][i][0], "volume": ob["asks"][i][1]}
                    for i in range(len(ob["asks"]))
                    if ob["asks"][i][1] > avg_ask_vol * wall_threshold
                ]
                
                # Añadir a la lista
                if bid_walls or ask_walls:
                    liquidity_walls.append({
                        "timestamp": ob.get("timestamp", time.time()),
                        "bid_walls": bid_walls,
                        "ask_walls": ask_walls
                    })
        
        # Guardar resultados
        result = {
            "timestamp": datetime.now().isoformat(),
            "avg_spread": avg_spread,
            "avg_bid_depth": avg_bid_depth,
            "avg_ask_depth": avg_ask_depth,
            "depth_ratio": depth_ratio,
            "top_level_volatility": top_level_volatility,
            "avg_bid_impact": avg_bid_impact,
            "avg_ask_impact": avg_ask_impact,
            "liquidity_walls": len(liquidity_walls),
            "book_symmetry": depth_ratio if depth_ratio <= 1 else 1/depth_ratio  # Normalización de simetría
        }
        
        # Actualizar resultados
        self.analysis_results[symbol]["orderbook_microstructure"] = result
        
        # Emitir eventos basados en el análisis
        
        # Evento de asimetría de liquidez
        asymmetry_threshold = 2.0  # 2:1 ratio es significativo
        if depth_ratio > asymmetry_threshold or depth_ratio < 1/asymmetry_threshold:
            direction = "bid" if depth_ratio > 1 else "ask"
            await self.emit_event("analysis.liquidity_asymmetry", {
                "symbol": symbol,
                "direction": direction,
                "ratio": depth_ratio if direction == "bid" else 1/depth_ratio,
                "timestamp": datetime.now().isoformat()
            })
            self.logger.info(f"Asimetría de liquidez detectada en {symbol}: {direction} {depth_ratio:.2f}x")
        
        # Evento de pared de liquidez
        if liquidity_walls:
            # Agrupar por nivel de precio
            all_walls = []
            for wall_data in liquidity_walls:
                all_walls.extend(
                    [{"side": "bid", **wall} for wall in wall_data["bid_walls"]] +
                    [{"side": "ask", **wall} for wall in wall_data["ask_walls"]]
                )
            
            # Encontrar las paredes más significativas
            if all_walls:
                biggest_wall = max(all_walls, key=lambda x: x["volume"])
                
                await self.emit_event("analysis.liquidity_wall", {
                    "symbol": symbol,
                    "side": biggest_wall["side"],
                    "price": biggest_wall["price"],
                    "volume": biggest_wall["volume"],
                    "wall_count": len(all_walls),
                    "timestamp": datetime.now().isoformat()
                })
                
                self.logger.info(
                    f"Pared de liquidez detectada en {symbol}: {biggest_wall['side']} "
                    f"a {biggest_wall['price']} ({biggest_wall['volume']})"
                )
    
    async def get_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener resultados de análisis para un símbolo.
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Resultados de análisis
        """
        if symbol not in self.analysis_results:
            return {}
        
        # Combinar resultados
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # Añadir análisis de precio
        if "price_action" in self.analysis_results[symbol]:
            pa = self.analysis_results[symbol]["price_action"]
            result.update({
                "volatility": pa.get("volatility"),
                "autocorrelation": pa.get("autocorrelation"),
                "is_mean_reverting": pa.get("is_mean_reverting"),
                "is_momentum": pa.get("is_momentum")
            })
        
        # Añadir análisis de trades
        if "trades_microstructure" in self.analysis_results[symbol]:
            tm = self.analysis_results[symbol]["trades_microstructure"]
            result.update({
                "trade_frequency": tm.get("trade_frequency"),
                "aggressive_ratio": tm.get("aggressive_ratio"),
                "iceberg_patterns": tm.get("iceberg_patterns")
            })
        
        # Añadir análisis de OrderBook
        if "orderbook_microstructure" in self.analysis_results[symbol]:
            om = self.analysis_results[symbol]["orderbook_microstructure"]
            result.update({
                "avg_spread": om.get("avg_spread"),
                "depth_ratio": om.get("depth_ratio"),
                "book_symmetry": om.get("book_symmetry")
            })
        
        # Calcular métricas compuestas
        
        # 1. Liquidez general
        if ("orderbook_microstructure" in self.analysis_results[symbol] and
                "trades_microstructure" in self.analysis_results[symbol]):
            om = self.analysis_results[symbol]["orderbook_microstructure"]
            tm = self.analysis_results[symbol]["trades_microstructure"]
            
            # Combinar profundidad de libro y frecuencia de trades
            total_depth = om.get("avg_bid_depth", 0) + om.get("avg_ask_depth", 0)
            liquidity_score = (total_depth / 1000) * (tm.get("trade_frequency", 0) / 10)
            
            result["liquidity_score"] = liquidity_score
        
        # 2. Régimen de mercado compuesto
        if "price_action" in self.analysis_results[symbol]:
            pa = self.analysis_results[symbol]["price_action"]
            
            if pa.get("is_mean_reverting"):
                result["market_regime"] = "mean_reversion"
                result["regime_strength"] = abs(pa.get("autocorrelation", 0))
            elif pa.get("is_momentum"):
                result["market_regime"] = "momentum"
                result["regime_strength"] = abs(pa.get("autocorrelation", 0))
            else:
                result["market_regime"] = "random_walk"
                result["regime_strength"] = 0
        
        # 3. Calidad del precio
        if ("orderbook_microstructure" in self.analysis_results[symbol] and
                "price_action" in self.analysis_results[symbol]):
            om = self.analysis_results[symbol]["orderbook_microstructure"]
            pa = self.analysis_results[symbol]["price_action"]
            
            # Combinar spread y volatilidad
            spread = om.get("avg_spread", 0)
            volatility = pa.get("volatility", 0)
            
            if volatility > 0:
                price_quality = 1 - (spread / volatility)
                result["price_quality"] = max(0, min(1, price_quality))
        
        # 4. Indicador de manipulación
        if all(key in self.analysis_results[symbol] for key in ["price_action", "trades_microstructure", "orderbook_microstructure"]):
            pa = self.analysis_results[symbol]["price_action"]
            tm = self.analysis_results[symbol]["trades_microstructure"]
            om = self.analysis_results[symbol]["orderbook_microstructure"]
            
            # Factores que sugieren manipulación
            manipulation_factors = [
                tm.get("has_flash_pattern", False),  # Flash crashes/rallies
                tm.get("iceberg_patterns", 0) > 2,    # Múltiples icebergs
                om.get("depth_ratio", 1) > 3 or om.get("depth_ratio", 1) < 0.33,  # Fuerte asimetría
                pa.get("volatility", 0) > 0.05,  # Alta volatilidad
                tm.get("aggressive_ratio", 0.5) > 0.8  # Agresividad extrema
            ]
            
            manipulation_score = sum(1 for factor in manipulation_factors if factor) / len(manipulation_factors)
            result["manipulation_score"] = manipulation_score
        
        return result
    
    async def generate_report(self, symbol: str) -> Dict[str, Any]:
        """
        Generar reporte completo de microestructura para un símbolo.
        
        Args:
            symbol: Símbolo a analizar
            
        Returns:
            Reporte detallado
        """
        # Obtener análisis básico
        analysis = await self.get_analysis(symbol)
        if not analysis:
            return {"error": "No hay datos suficientes para el análisis"}
        
        # Generar gráficos
        chart_path = await self._generate_chart(symbol)
        
        # Añadir insights
        insights = []
        
        # Análisis de régimen de mercado
        if "market_regime" in analysis:
            regime = analysis["market_regime"]
            strength = analysis.get("regime_strength", 0)
            
            if regime == "mean_reversion":
                insights.append(f"Mercado en régimen de reversión a la media (fuerza: {strength:.2f}). Considerar estrategias contrarian.")
            elif regime == "momentum":
                insights.append(f"Mercado en régimen de momentum (fuerza: {strength:.2f}). Considerar estrategias de seguimiento de tendencia.")
            else:
                insights.append("Mercado en régimen de paseo aleatorio. Sin patrón claro.")
        
        # Análisis de liquidez
        if "liquidity_score" in analysis:
            liquidity = analysis["liquidity_score"]
            if liquidity > 5:
                insights.append(f"Alta liquidez (score: {liquidity:.2f}). Adecuado para posiciones grandes.")
            elif liquidity > 1:
                insights.append(f"Liquidez media (score: {liquidity:.2f}). Precaución con posiciones grandes.")
            else:
                insights.append(f"Baja liquidez (score: {liquidity:.2f}). Evitar posiciones grandes.")
        
        # Análisis de manipulación
        if "manipulation_score" in analysis:
            manipulation = analysis["manipulation_score"]
            if manipulation > 0.6:
                insights.append(f"Alta probabilidad de manipulación (score: {manipulation:.2f}). Precaución extrema.")
            elif manipulation > 0.3:
                insights.append(f"Posible manipulación (score: {manipulation:.2f}). Observar con atención.")
        
        # Análisis de calidad de precio
        if "price_quality" in analysis:
            price_quality = analysis["price_quality"]
            if price_quality > 0.8:
                insights.append(f"Alta calidad de precio (score: {price_quality:.2f}). Spreads ajustados relativos a la volatilidad.")
            elif price_quality < 0.3:
                insights.append(f"Baja calidad de precio (score: {price_quality:.2f}). Spreads amplios relativos a la volatilidad.")
        
        # Añadir al reporte
        analysis["insights"] = insights
        analysis["chart_path"] = chart_path
        
        return analysis
    
    async def _generate_chart(self, symbol: str) -> str:
        """
        Generar gráfico de análisis de microestructura.
        
        Args:
            symbol: Símbolo a graficar
            
        Returns:
            Ruta al archivo del gráfico
        """
        # Verificar si hay datos suficientes
        if symbol not in self.analysis_results:
            return ""
        
        try:
            # Crear figura
            plt.figure(figsize=(12, 10))
            
            # Datos de precio
            if "price_action" in self.analysis_results[symbol]:
                pa = self.analysis_results[symbol]["price_action"]
                
                # Subplot para autocorrelación
                plt.subplot(3, 2, 1)
                plt.bar(["Lag 1"], [pa.get("autocorrelation", 0)])
                plt.axhline(y=0.2, color='g', linestyle='--')
                plt.axhline(y=-0.2, color='r', linestyle='--')
                plt.title("Autocorrelación de Retornos")
                
                # Subplot para volumen/volatilidad
                plt.subplot(3, 2, 2)
                plt.bar(["Vol. Clustering"], [pa.get("volatility_clustering", 0)])
                plt.title("Clustering de Volatilidad")
                plt.ylim(0, 1)
            
            # Datos de trades
            if "trades_microstructure" in self.analysis_results[symbol]:
                tm = self.analysis_results[symbol]["trades_microstructure"]
                
                # Subplot para ratio de agresividad
                plt.subplot(3, 2, 3)
                plt.pie([tm.get("aggressive_ratio", 0), 1 - tm.get("aggressive_ratio", 0)], 
                       labels=["Agresivos", "Pasivos"],
                       autopct='%1.1f%%')
                plt.title("Composición de Trades")
                
                # Subplot para icebergs
                plt.subplot(3, 2, 4)
                plt.bar(["Icebergs"], [tm.get("iceberg_patterns", 0)])
                plt.title("Patrones de Iceberg Detectados")
            
            # Datos de OrderBook
            if "orderbook_microstructure" in self.analysis_results[symbol]:
                om = self.analysis_results[symbol]["orderbook_microstructure"]
                
                # Subplot para profundidad
                plt.subplot(3, 2, 5)
                plt.bar(["Compra", "Venta"], [om.get("avg_bid_depth", 0), om.get("avg_ask_depth", 0)])
                plt.title("Profundidad del Libro")
                
                # Subplot para spread
                plt.subplot(3, 2, 6)
                plt.bar(["Spread"], [om.get("avg_spread", 0) * 100])  # Mostrar como porcentaje
                plt.title("Spread Promedio (%)")
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.plot_dir}/{symbol}_microstructure_{timestamp}.png"
            plt.savefig(file_path)
            plt.close()
            
            return file_path
        
        except Exception as e:
            self.logger.error(f"Error al generar gráfico: {e}")
            return ""


# Funciones auxiliares para análisis de microestructura

def calculate_vpin(
    volume_series: np.ndarray,
    price_series: np.ndarray,
    bucket_size: int = 50
) -> float:
    """
    Calcular Volume-Synchronized Probability of Informed Trading (VPIN).
    
    VPIN es una medida del flujo tóxico de órdenes, que indica la presencia
    de traders informados en el mercado.
    
    Args:
        volume_series: Serie de volúmenes
        price_series: Serie de precios
        bucket_size: Tamaño del bucket para normalización
        
    Returns:
        Valor VPIN entre 0 y 1
    """
    if len(volume_series) != len(price_series) or len(volume_series) < 2:
        return 0.0
    
    # Calcular cambios de precio
    price_changes = np.diff(price_series)
    
    # Separar volumen en volumen de "compra" y "venta" basado en el cambio de precio
    buy_volume = np.zeros(len(price_changes))
    sell_volume = np.zeros(len(price_changes))
    
    for i in range(len(price_changes)):
        if price_changes[i] > 0:
            # Cambio positivo -> más volumen de compra
            buy_volume[i] = volume_series[i+1] * (price_changes[i] / abs(price_changes[i]))
            sell_volume[i] = 0
        elif price_changes[i] < 0:
            # Cambio negativo -> más volumen de venta
            buy_volume[i] = 0
            sell_volume[i] = volume_series[i+1] * (abs(price_changes[i]) / abs(price_changes[i]))
        else:
            # Sin cambio -> igual volumen de compra y venta
            buy_volume[i] = volume_series[i+1] / 2
            sell_volume[i] = volume_series[i+1] / 2
    
    # Calcular VPIN como promedio del valor absoluto de desbalance
    total_volume = np.sum(buy_volume + sell_volume)
    if total_volume == 0:
        return 0.0
    
    imbalance = np.abs(buy_volume - sell_volume)
    vpin = np.sum(imbalance) / total_volume
    
    return vpin


def estimate_kyle_lambda(
    price_changes: np.ndarray,
    volume_series: np.ndarray
) -> float:
    """
    Estimar el parámetro lambda de Kyle, que mide el impacto de precio.
    
    Args:
        price_changes: Serie de cambios de precio
        volume_series: Serie de volúmenes
        
    Returns:
        Lambda de Kyle (impacto de precio por unidad de volumen)
    """
    if len(price_changes) != len(volume_series) or len(price_changes) < 2:
        return 0.0
    
    # Calcular regresión lineal de cambio de precio vs volumen
    try:
        # Usar mínimos cuadrados para estimar lambda
        lambda_est = np.sum(price_changes * volume_series) / np.sum(volume_series ** 2)
        return lambda_est
    except Exception:
        return 0.0


def calculate_flow_toxicity(
    price_series: np.ndarray,
    volume_series: np.ndarray,
    window: int = 50
) -> np.ndarray:
    """
    Calcular toxicidad del flujo de órdenes.
    
    La toxicidad alta indica presencia de traders informados.
    
    Args:
        price_series: Serie de precios
        volume_series: Serie de volúmenes
        window: Tamaño de la ventana móvil
        
    Returns:
        Serie de toxicidad
    """
    if len(price_series) != len(volume_series) or len(price_series) < window:
        return np.array([])
    
    toxicity = np.zeros(len(price_series) - window + 1)
    
    for i in range(len(toxicity)):
        price_window = price_series[i:i+window]
        volume_window = volume_series[i:i+window]
        
        # Calcular VPIN en la ventana
        vpin = calculate_vpin(volume_window, price_window)
        
        # Estimar lambda de Kyle
        price_changes = np.diff(price_window)
        vol_window_adj = volume_window[1:]
        kyle_lambda = estimate_kyle_lambda(price_changes, vol_window_adj)
        
        # Combinar métricas para toxicidad
        # VPIN alto y lambda alto indican mayor toxicidad
        toxicity[i] = vpin * abs(kyle_lambda)
    
    return toxicity


# Exportación para uso fácil
market_microstructure_analyzer = MarketMicrostructureAnalyzer()