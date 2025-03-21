"""
Módulo de análisis de microestructura de mercado.

Este módulo proporciona herramientas para analizar el comportamiento
de mercado a nivel micro, incluyendo flujo de órdenes, profundidad de mercado,
impacto en el precio y liquidez.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

class MarketMicrostructureAnalyzer:
    """
    Analizador de microestructura de mercado.
    
    Esta clase proporciona métodos para analizar patrones en el flujo de órdenes,
    la profundidad del mercado, la liquidez, y otros aspectos de la microestructura
    del mercado.
    """
    
    def __init__(self):
        """Inicializar el analizador de microestructura."""
        pass
    
    def calculate_price_impact(self, orderbook: Dict[str, List[List[float]]]) -> Dict[str, float]:
        """
        Calcula el impacto de mercado de órdenes de diferentes tamaños.
        
        Args:
            orderbook: Libro de órdenes con bids y asks
            
        Returns:
            Diccionario con impactos para diferentes tamaños
        """
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            return {}
            
        bids = orderbook["bids"]  # Lista de [precio, cantidad]
        asks = orderbook["asks"]  # Lista de [precio, cantidad]
        
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        # Calcular impacto para diferentes tamaños de órdenes
        impacts = {}
        for size in [1, 5, 10, 20, 50, 100]:  # Diferentes tamaños de órdenes (unidades base)
            # Impacto de compra (slippage por vender)
            sell_impact = self._calculate_slippage(bids, size, "sell", mid_price)
            
            # Impacto de venta (slippage por comprar)
            buy_impact = self._calculate_slippage(asks, size, "buy", mid_price)
            
            impacts[f"impact_{size}"] = {
                "buy": buy_impact,
                "sell": sell_impact,
                "total": (buy_impact + sell_impact) / 2
            }
            
        return impacts
    
    def _calculate_slippage(
        self, 
        orders: List[List[float]], 
        size: float, 
        side: str, 
        mid_price: float
    ) -> float:
        """
        Calcula el slippage para un tamaño y lado específicos.
        
        Args:
            orders: Lista de órdenes [precio, cantidad]
            size: Tamaño de la orden
            side: Lado ('buy' o 'sell')
            mid_price: Precio medio actual
            
        Returns:
            Slippage como porcentaje
        """
        remaining_size = size
        executed_value = 0.0
        
        for order in orders:
            price, amount = order
            
            if remaining_size <= 0:
                break
                
            executed_amount = min(amount, remaining_size)
            executed_value += executed_amount * price
            remaining_size -= executed_amount
            
        if size - remaining_size <= 0:
            return 0.0
            
        # Precio promedio de ejecución
        avg_execution_price = executed_value / (size - remaining_size)
        
        # Calcular slippage como porcentaje
        if side == "buy":
            slippage = (avg_execution_price - mid_price) / mid_price
        else:  # sell
            slippage = (mid_price - avg_execution_price) / mid_price
            
        return slippage * 100  # Convertir a porcentaje
    
    def calculate_market_depth(self, orderbook: Dict[str, List[List[float]]]) -> Dict[str, float]:
        """
        Calcula la profundidad del mercado a diferentes niveles de precio.
        
        Args:
            orderbook: Libro de órdenes con bids y asks
            
        Returns:
            Métricas de profundidad de mercado
        """
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            return {}
            
        bids = orderbook["bids"]  # Lista de [precio, cantidad]
        asks = orderbook["asks"]  # Lista de [precio, cantidad]
        
        # Precio medio
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        # Spread
        spread = asks[0][0] - bids[0][0]
        spread_pct = spread / mid_price * 100
        
        # Volumen acumulado a diferentes niveles
        bid_volumes = {}
        ask_volumes = {}
        
        for level in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:  # Porcentajes de precio
            price_level = level / 100  # Convertir a fracción
            
            # Volumen acumulado de compra dentro del nivel
            bid_vol = sum(order[1] for order in bids if order[0] >= mid_price * (1 - price_level))
            
            # Volumen acumulado de venta dentro del nivel
            ask_vol = sum(order[1] for order in asks if order[0] <= mid_price * (1 + price_level))
            
            bid_volumes[f"bid_depth_{level}pct"] = bid_vol
            ask_volumes[f"ask_depth_{level}pct"] = ask_vol
            
        # Combinar resultados
        result = {
            "mid_price": mid_price,
            "spread": spread,
            "spread_pct": spread_pct,
            "best_bid": bids[0][0],
            "best_ask": asks[0][0],
            "best_bid_volume": bids[0][1],
            "best_ask_volume": asks[0][1],
            "bid_ask_ratio": bids[0][1] / asks[0][1] if asks[0][1] > 0 else float('inf'),
            **bid_volumes,
            **ask_volumes
        }
        
        return result
    
    def analyze_trade_flow(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza el flujo de transacciones para detectar patrones.
        
        Args:
            trades: Lista de operaciones realizadas
            
        Returns:
            Análisis del flujo de operaciones
        """
        if not trades:
            return {}
            
        # Convertir a DataFrame para facilitar el análisis
        df = pd.DataFrame(trades)
        
        # Asegurarse de que tenemos las columnas necesarias
        required_columns = ["price", "amount", "side", "timestamp"]
        for col in required_columns:
            if col not in df.columns:
                return {}
                
        # Asegurarse de que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        # Volumen total
        buy_volume = df[df["side"] == "buy"]["amount"].sum()
        sell_volume = df[df["side"] == "sell"]["amount"].sum()
        total_volume = buy_volume + sell_volume
        
        # Calcular valor transaccionado
        df["value"] = df["price"] * df["amount"]
        buy_value = df[df["side"] == "buy"]["value"].sum()
        sell_value = df[df["side"] == "sell"]["value"].sum()
        total_value = buy_value + sell_value
        
        # Calcular precio promedio ponderado por volumen (VWAP)
        vwap = total_value / total_volume if total_volume > 0 else 0
        
        # Contar número de operaciones por lado
        buy_count = len(df[df["side"] == "buy"])
        sell_count = len(df[df["side"] == "sell"])
        trade_count = len(df)
        
        # Calcular tamaño promedio de operación
        avg_trade_size = total_volume / trade_count if trade_count > 0 else 0
        avg_buy_size = buy_volume / buy_count if buy_count > 0 else 0
        avg_sell_size = sell_volume / sell_count if sell_count > 0 else 0
        
        # Desequilibrio de flujo de órdenes (Order Flow Imbalance)
        ofi = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
        
        # Tasa de llegada de órdenes (trades per minute)
        time_range = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60
        trade_rate = trade_count / time_range if time_range > 0 else 0
        
        return {
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "total_value": total_value,
            "vwap": vwap,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "trade_count": trade_count,
            "avg_trade_size": avg_trade_size,
            "avg_buy_size": avg_buy_size,
            "avg_sell_size": avg_sell_size,
            "buy_sell_ratio": buy_volume / sell_volume if sell_volume > 0 else float('inf'),
            "order_flow_imbalance": ofi,
            "trade_rate_per_minute": trade_rate
        }
    
    def estimate_market_impact(
        self, 
        orderbook: Dict[str, List[List[float]]], 
        order_size: float, 
        side: str
    ) -> Dict[str, float]:
        """
        Estima el impacto de mercado para una orden de tamaño determinado.
        
        Args:
            orderbook: Libro de órdenes con bids y asks
            order_size: Tamaño de la orden a evaluar
            side: Lado de la orden ('buy' o 'sell')
            
        Returns:
            Métricas de impacto estimado
        """
        if not orderbook or side not in ["buy", "sell"]:
            return {}
            
        book_side = "asks" if side == "buy" else "bids"
        if book_side not in orderbook or not orderbook[book_side]:
            return {}
            
        orders = orderbook[book_side]
        best_price = orders[0][0]
        
        # Calcular ejecución
        remaining_size = order_size
        executed_value = 0.0
        levels_used = 0
        
        for i, order in enumerate(orders):
            price, amount = order
            
            if remaining_size <= 0:
                break
                
            executed_amount = min(amount, remaining_size)
            executed_value += executed_amount * price
            remaining_size -= executed_amount
            levels_used = i + 1
            
        # Si no se pudo ejecutar toda la orden
        if remaining_size > 0:
            unfilled_ratio = remaining_size / order_size
            avg_price = executed_value / (order_size - remaining_size) if order_size > remaining_size else 0
        else:
            unfilled_ratio = 0.0
            avg_price = executed_value / order_size
            
        # Calcular impacto de precio
        price_impact = ((avg_price - best_price) / best_price) * 100 if side == "buy" else \
                      ((best_price - avg_price) / best_price) * 100
                      
        return {
            "average_price": avg_price,
            "best_price": best_price,
            "price_impact_pct": price_impact,
            "unfilled_ratio": unfilled_ratio,
            "levels_used": levels_used,
            "is_marketable": unfilled_ratio < 0.1  # Si se puede ejecutar >90% de la orden
        }
    
    def analyze_liquidity(self, orderbook_time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza la evolución de la liquidez a lo largo del tiempo.
        
        Args:
            orderbook_time_series: Serie temporal de libros de órdenes
            
        Returns:
            Métricas de liquidez en el tiempo
        """
        if not orderbook_time_series:
            return {}
            
        spreads = []
        bid_volumes = []
        ask_volumes = []
        timestamps = []
        
        for entry in orderbook_time_series:
            if "timestamp" not in entry or "orderbook" not in entry:
                continue
                
            orderbook = entry["orderbook"]
            if "bids" not in orderbook or "asks" not in orderbook:
                continue
                
            bids = orderbook["bids"]
            asks = orderbook["asks"]
            
            if not bids or not asks:
                continue
                
            # Calcular spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_pct = spread / mid_price * 100
            
            # Calcular volumen acumulado (profundidad a 1%)
            bid_volume = sum(order[1] for order in bids if order[0] >= mid_price * 0.99)
            ask_volume = sum(order[1] for order in asks if order[0] <= mid_price * 1.01)
            
            # Almacenar
            spreads.append(spread_pct)
            bid_volumes.append(bid_volume)
            ask_volumes.append(ask_volume)
            timestamps.append(entry["timestamp"])
            
        if not spreads:
            return {}
            
        # Calcular estadísticas
        avg_spread = np.mean(spreads)
        min_spread = np.min(spreads)
        max_spread = np.max(spreads)
        spread_volatility = np.std(spreads)
        
        avg_bid_volume = np.mean(bid_volumes)
        avg_ask_volume = np.mean(ask_volumes)
        
        # Calcular resiliencia (qué tan rápido se recupera el libro tras cambios)
        spread_autocorr = np.corrcoef(spreads[:-1], spreads[1:])[0, 1] if len(spreads) > 1 else 0
        
        # Calcular tendencia
        if len(spreads) > 10:
            # Ajuste lineal simple para ver tendencia
            xs = np.arange(len(spreads))
            spread_slope = np.polyfit(xs, spreads, 1)[0]
            volume_slope = np.polyfit(xs, bid_volumes + ask_volumes, 1)[0]
        else:
            spread_slope = 0
            volume_slope = 0
            
        return {
            "avg_spread_pct": avg_spread,
            "min_spread_pct": min_spread,
            "max_spread_pct": max_spread,
            "spread_volatility": spread_volatility,
            "avg_bid_volume": avg_bid_volume,
            "avg_ask_volume": avg_ask_volume,
            "avg_total_volume": avg_bid_volume + avg_ask_volume,
            "liquidity_imbalance": (avg_bid_volume - avg_ask_volume) / (avg_bid_volume + avg_ask_volume)
                                if (avg_bid_volume + avg_ask_volume) > 0 else 0,
            "spread_autocorrelation": spread_autocorr,
            "spread_trend": "decreasing" if spread_slope < -0.01 else "increasing" if spread_slope > 0.01 else "stable",
            "volume_trend": "decreasing" if volume_slope < -1 else "increasing" if volume_slope > 1 else "stable",
            "timestamps": timestamps,
            "spreads": spreads,
            "bid_volumes": bid_volumes,
            "ask_volumes": ask_volumes
        }