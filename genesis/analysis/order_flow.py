"""
Módulo de análisis de flujo de órdenes.

Este módulo proporciona herramientas para analizar patrones en el
flujo de órdenes, incluyendo desequilibrios, dominancia de compra/venta,
y actividad inusual.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

class OrderFlowAnalyzer:
    """
    Analizador de flujo de órdenes.
    
    Esta clase proporciona métodos para analizar y detectar patrones
    en el flujo de órdenes, incluyendo desequilibrios y actividad inusual
    que podría indicar cambios en la dirección del mercado.
    """
    
    def __init__(self):
        """Inicializar el analizador de flujo de órdenes."""
        pass
    
    def calculate_voi(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcular el Índice de Desequilibrio de Volumen (VOI - Volume Imbalance Index).
        
        Este indicador mide la presión de compra/venta basándose en el volumen
        negociado a diferentes niveles de precio.
        
        Args:
            trades: Lista de operaciones con precio, cantidad y lado
            
        Returns:
            Métricas de desequilibrio de volumen
        """
        if not trades:
            return {}
            
        # Convertir a DataFrame
        df = pd.DataFrame(trades)
        
        # Verificar columnas necesarias
        required_columns = ["price", "amount", "side", "timestamp"]
        for col in required_columns:
            if col not in df.columns:
                return {}
                
        # Asegurarse de que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        # Calcular delta VOI para cada operación
        df["delta_voi"] = df.apply(
            lambda x: x["amount"] if x["side"] == "buy" else -x["amount"],
            axis=1
        )
        
        # Agrupar por intervalos de tiempo (5 minutos)
        df["time_group"] = df["timestamp"].dt.floor("5min")
        
        # Calcular VOI acumulado para cada grupo de tiempo
        voi_by_time = df.groupby("time_group")["delta_voi"].sum().reset_index()
        voi_by_time.columns = ["timestamp", "voi"]
        
        # Calcular VOI acumulado total
        cumulative_voi = df["delta_voi"].cumsum().tolist()
        
        # Calcular métricas
        total_voi = df["delta_voi"].sum()
        voi_last_1h = df[df["timestamp"] > df["timestamp"].max() - timedelta(hours=1)]["delta_voi"].sum()
        
        # Calcular tendencia
        if len(cumulative_voi) > 10:
            # Últimos 10 intervalos
            recent_trend = cumulative_voi[-1] - cumulative_voi[-10]
            trend = "bullish" if recent_trend > 0 else "bearish" if recent_trend < 0 else "neutral"
        else:
            trend = "neutral"
            
        return {
            "voi_total": total_voi,
            "voi_last_1h": voi_last_1h,
            "voi_by_time": voi_by_time.to_dict("records"),
            "cumulative_voi": cumulative_voi,
            "trend": trend
        }
    
    def calculate_cvd(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcular el Delta de Volumen Cumulativo (CVD - Cumulative Volume Delta).
        
        Este indicador mide el desequilibrio entre volumen de compra y venta,
        y se utiliza para identificar presión de compra/venta.
        
        Args:
            trades: Lista de operaciones con precio, cantidad y lado
            
        Returns:
            Métricas de delta de volumen acumulado
        """
        if not trades:
            return {}
            
        # Convertir a DataFrame
        df = pd.DataFrame(trades)
        
        # Verificar columnas necesarias
        required_columns = ["price", "amount", "side", "timestamp"]
        for col in required_columns:
            if col not in df.columns:
                return {}
                
        # Asegurarse de que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        # Calcular delta para cada operación
        df["delta"] = df.apply(
            lambda x: x["amount"] if x["side"] == "buy" else -x["amount"],
            axis=1
        )
        
        # Calcular CVD acumulado
        df["cvd"] = df["delta"].cumsum()
        
        # Calcular estadísticas
        cvd_series = df["cvd"].tolist()
        timestamps = df["timestamp"].tolist()
        
        # Calcular pendiente reciente (últimas 20 operaciones o todas si hay menos)
        window_size = min(20, len(df))
        if window_size > 1:
            recent_slope = (cvd_series[-1] - cvd_series[-window_size]) / window_size
        else:
            recent_slope = 0
            
        # Calcular otros estadísticos
        max_cvd = df["cvd"].max()
        min_cvd = df["cvd"].min()
        latest_cvd = df["cvd"].iloc[-1] if not df.empty else 0
        
        # Obtener puntos de cambio significativos (donde la tendencia se invierte)
        sign_changes = []
        prev_sign = 0
        
        for i in range(1, len(df)):
            delta = df["delta"].iloc[i]
            curr_sign = 1 if delta > 0 else -1 if delta < 0 else 0
            
            if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
                sign_changes.append({
                    "timestamp": df["timestamp"].iloc[i],
                    "price": df["price"].iloc[i],
                    "cvd": df["cvd"].iloc[i],
                    "direction": "bullish" if curr_sign > 0 else "bearish"
                })
                
            if curr_sign != 0:
                prev_sign = curr_sign
        
        return {
            "cvd_values": cvd_series,
            "timestamps": timestamps,
            "max_cvd": max_cvd,
            "min_cvd": min_cvd,
            "latest_cvd": latest_cvd,
            "recent_slope": recent_slope,
            "trend": "bullish" if recent_slope > 0 else "bearish" if recent_slope < 0 else "neutral",
            "sign_changes": sign_changes
        }
    
    def detect_large_orders(
        self, 
        trades: List[Dict[str, Any]], 
        threshold_multiplier: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detectar operaciones grandes que puedan indicar actividad inusual.
        
        Args:
            trades: Lista de operaciones con precio, cantidad y lado
            threshold_multiplier: Multiplicador para considerar una operación "grande"
            
        Returns:
            Lista de operaciones grandes detectadas
        """
        if not trades:
            return []
            
        # Convertir a DataFrame
        df = pd.DataFrame(trades)
        
        # Verificar columnas necesarias
        required_columns = ["price", "amount", "side", "timestamp"]
        for col in required_columns:
            if col not in df.columns:
                return []
                
        # Calcular tamaño promedio
        avg_size = df["amount"].mean()
        
        # Detectar operaciones grandes
        large_trades = df[df["amount"] > avg_size * threshold_multiplier].copy()
        
        # Ordenar por tamaño descendente
        large_trades = large_trades.sort_values("amount", ascending=False)
        
        # Calcular el impacto en el precio
        if len(large_trades) > 0 and len(df) > 0:
            for i, row in large_trades.iterrows():
                # Encontrar el índice de esta operación en el DataFrame original
                idx = df.index.get_loc(i)
                
                # Precio antes y después (si es posible)
                if idx > 0:
                    price_before = df.iloc[idx-1]["price"]
                else:
                    price_before = row["price"]
                    
                if idx < len(df) - 1:
                    price_after = df.iloc[idx+1]["price"]
                else:
                    price_after = row["price"]
                    
                # Calcular el impacto
                large_trades.loc[i, "price_impact_pct"] = ((price_after - price_before) / price_before) * 100
        
        # Convertir a lista de diccionarios
        result = large_trades.to_dict("records")
        
        # Añadir contexto
        for i, trade in enumerate(result):
            trade["size_vs_avg"] = trade["amount"] / avg_size
            trade["rank"] = i + 1
            
        return result
    
    def analyze_trade_clusters(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analizar clústeres de operaciones que puedan indicar comportamiento coordinado.
        
        Args:
            trades: Lista de operaciones con precio, cantidad y lado
            
        Returns:
            Análisis de clústeres de operaciones
        """
        if not trades:
            return {}
            
        # Convertir a DataFrame
        df = pd.DataFrame(trades)
        
        # Verificar columnas necesarias
        required_columns = ["price", "amount", "side", "timestamp"]
        for col in required_columns:
            if col not in df.columns:
                return {}
                
        # Asegurarse de que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Ordenar por timestamp
        df = df.sort_values("timestamp")
        
        # Calcular tiempo entre operaciones
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
        
        # Definir un clúster como operaciones separadas por menos de 1 segundo
        df["cluster_start"] = df["time_diff"] > 1.0
        df["cluster_id"] = df["cluster_start"].cumsum()
        
        # Analizar clústeres
        clusters = []
        
        for cluster_id, group in df.groupby("cluster_id"):
            if len(group) < 2:
                continue  # Ignorar operaciones solitarias
                
            # Calcular propiedades del clúster
            duration = (group["timestamp"].max() - group["timestamp"].min()).total_seconds()
            buy_count = (group["side"] == "buy").sum()
            sell_count = (group["side"] == "sell").sum()
            buy_volume = group[group["side"] == "buy"]["amount"].sum()
            sell_volume = group[group["side"] == "sell"]["amount"].sum()
            
            # Determinar la dirección del clúster
            if buy_volume > sell_volume:
                direction = "buy"
            elif sell_volume > buy_volume:
                direction = "sell"
            else:
                direction = "neutral"
                
            # Calcular el cambio de precio durante el clúster
            price_start = group["price"].iloc[0]
            price_end = group["price"].iloc[-1]
            price_change = price_end - price_start
            price_change_pct = (price_change / price_start) * 100 if price_start > 0 else 0
            
            clusters.append({
                "cluster_id": cluster_id,
                "start_time": group["timestamp"].min(),
                "end_time": group["timestamp"].max(),
                "duration_sec": duration,
                "trade_count": len(group),
                "trades_per_second": len(group) / max(duration, 0.001),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "total_volume": buy_volume + sell_volume,
                "price_start": price_start,
                "price_end": price_end,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "direction": direction
            })
            
        # Filtrar clústeres significativos (alta velocidad de operaciones o gran volumen)
        if clusters:
            avg_volume = sum(c["total_volume"] for c in clusters) / len(clusters)
            avg_trades_per_second = sum(c["trades_per_second"] for c in clusters) / len(clusters)
            
            significant_clusters = [
                c for c in clusters if 
                c["trades_per_second"] > avg_trades_per_second * 1.5 or
                c["total_volume"] > avg_volume * 1.5
            ]
        else:
            significant_clusters = []
            
        return {
            "total_clusters": len(clusters),
            "significant_clusters": len(significant_clusters),
            "clusters": clusters,
            "significant_clusters_detail": significant_clusters
        }
    
    def calculate_liquidity_consumption(
        self, 
        trades: List[Dict[str, Any]], 
        orderbook_before_after: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calcular el consumo de liquidez, indicando la agresividad de las órdenes.
        
        Args:
            trades: Lista de operaciones
            orderbook_before_after: Snapshots del libro de órdenes antes y después de operaciones grandes
            
        Returns:
            Métricas de consumo de liquidez
        """
        if not trades or not orderbook_before_after:
            return {}
            
        # Analizar cada evento de consumo de liquidez
        liquidity_events = []
        
        for entry in orderbook_before_after:
            if "trade" not in entry or "orderbook_before" not in entry or "orderbook_after" not in entry:
                continue
                
            trade = entry["trade"]
            book_before = entry["orderbook_before"]
            book_after = entry["orderbook_after"]
            
            # Verificar datos necesarios
            if "side" not in trade or "amount" not in trade or "price" not in trade:
                continue
                
            side = trade["side"]
            amount = trade["amount"]
            price = trade["price"]
            book_side = "asks" if side == "buy" else "bids"
            
            # Calcular liquidez consumida
            if book_side in book_before and book_side in book_after:
                # Calcular liquidez disponible antes
                before_levels = book_before[book_side]
                after_levels = book_after[book_side]
                
                # Liquidez desaparecida
                if side == "buy":
                    # Para compras, se consume liquidez en los asks (ventas)
                    best_ask_before = before_levels[0][0] if before_levels else None
                    best_ask_after = after_levels[0][0] if after_levels else None
                    
                    if best_ask_before is not None and best_ask_after is not None:
                        # Verificar si se consumió un nivel completo
                        levels_consumed = 0
                        volume_consumed = 0
                        
                        for level in before_levels:
                            if level[0] <= price:
                                # Este nivel fue ejecutado
                                levels_consumed += 1
                                volume_consumed += level[1]
                            else:
                                break
                                
                        liquidity_events.append({
                            "timestamp": trade.get("timestamp"),
                            "side": side,
                            "trade_amount": amount,
                            "trade_price": price,
                            "levels_consumed": levels_consumed,
                            "volume_consumed": volume_consumed,
                            "price_impact": ((best_ask_after - best_ask_before) / best_ask_before) * 100 
                                          if best_ask_before > 0 else 0,
                            "aggressive": volume_consumed > amount * 1.2  # Operación que consume más liquidez de la necesaria
                        })
                        
                else:  # sell
                    # Para ventas, se consume liquidez en los bids (compras)
                    best_bid_before = before_levels[0][0] if before_levels else None
                    best_bid_after = after_levels[0][0] if after_levels else None
                    
                    if best_bid_before is not None and best_bid_after is not None:
                        # Verificar si se consumió un nivel completo
                        levels_consumed = 0
                        volume_consumed = 0
                        
                        for level in before_levels:
                            if level[0] >= price:
                                # Este nivel fue ejecutado
                                levels_consumed += 1
                                volume_consumed += level[1]
                            else:
                                break
                                
                        liquidity_events.append({
                            "timestamp": trade.get("timestamp"),
                            "side": side,
                            "trade_amount": amount,
                            "trade_price": price,
                            "levels_consumed": levels_consumed,
                            "volume_consumed": volume_consumed,
                            "price_impact": ((best_bid_before - best_bid_after) / best_bid_before) * 100 
                                          if best_bid_before > 0 else 0,
                            "aggressive": volume_consumed > amount * 1.2  # Operación que consume más liquidez de la necesaria
                        })
        
        # Calcular estadísticas agregadas
        if not liquidity_events:
            return {"events": []}
            
        buy_events = [e for e in liquidity_events if e["side"] == "buy"]
        sell_events = [e for e in liquidity_events if e["side"] == "sell"]
        
        # Calcular promedios para eventos de compra
        if buy_events:
            avg_buy_levels = sum(e["levels_consumed"] for e in buy_events) / len(buy_events)
            avg_buy_impact = sum(e["price_impact"] for e in buy_events) / len(buy_events)
            aggressive_buys = sum(1 for e in buy_events if e.get("aggressive", False))
        else:
            avg_buy_levels = 0
            avg_buy_impact = 0
            aggressive_buys = 0
            
        # Calcular promedios para eventos de venta
        if sell_events:
            avg_sell_levels = sum(e["levels_consumed"] for e in sell_events) / len(sell_events)
            avg_sell_impact = sum(e["price_impact"] for e in sell_events) / len(sell_events)
            aggressive_sells = sum(1 for e in sell_events if e.get("aggressive", False))
        else:
            avg_sell_levels = 0
            avg_sell_impact = 0
            aggressive_sells = 0
            
        return {
            "events": liquidity_events,
            "buy_events_count": len(buy_events),
            "sell_events_count": len(sell_events),
            "avg_buy_levels_consumed": avg_buy_levels,
            "avg_sell_levels_consumed": avg_sell_levels,
            "avg_buy_price_impact": avg_buy_impact,
            "avg_sell_price_impact": avg_sell_impact,
            "aggressive_buys": aggressive_buys,
            "aggressive_sells": aggressive_sells,
            "buy_sell_aggression_ratio": aggressive_buys / aggressive_sells if aggressive_sells > 0 else float('inf')
        }