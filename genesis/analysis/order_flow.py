"""
Análisis de OrderFlow para el sistema Genesis.

Este módulo proporciona funcionalidades para analizar el flujo de órdenes
y detectar patrones de acumulación, distribución y desequilibrios en el mercado.
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


class OrderFlowAnalyzer(Component):
    """
    Analizador de OrderFlow.
    
    Este componente analiza datos de OrderBook y trades en tiempo real
    para detectar patrones de acumulación, distribución y desequilibrios.
    """
    
    def __init__(self, name: str = "order_flow"):
        """
        Inicializar el analizador de OrderFlow.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Configuración
        self.volume_threshold = 0.1  # 10% del volumen normal es considerado significativo
        self.imbalance_threshold = 0.7  # 70% de desequilibrio se considera significativo
        self.period = 100  # Número de ticks a analizar
        
        # Datos de OrderBook
        # Symbol -> (timestamp, book_data)
        self.order_books: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        
        # Datos de trades
        # Symbol -> (timestamp, trade_data)
        self.trades: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        
        # Resultados de análisis
        # Symbol -> análisis
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        
        # Directorio para gráficos
        self.plot_dir = "data/plots/order_flow"
        os.makedirs(self.plot_dir, exist_ok=True)
    
    async def start(self) -> None:
        """Iniciar el analizador de OrderFlow."""
        await super().start()
        self.logger.info("Analizador de OrderFlow iniciado")
    
    async def stop(self) -> None:
        """Detener el analizador de OrderFlow."""
        await super().stop()
        self.logger.info("Analizador de OrderFlow detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Procesar datos del OrderBook
        if event_type == "market.orderbook_updated":
            await self._process_orderbook(data)
        
        # Procesar datos de trades
        elif event_type == "market.trade_executed":
            await self._process_trade(data)
    
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
        self.order_books[symbol] = self.order_books[symbol][-self.period:]
        
        # Analizar si tenemos suficientes datos
        if len(self.order_books[symbol]) >= 10:  # Mínimo 10 muestras para análisis
            await self._analyze_orderbook(symbol)
    
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
        self.trades[symbol] = self.trades[symbol][-self.period:]
        
        # Analizar si tenemos suficientes datos
        if len(self.trades[symbol]) >= 10:  # Mínimo 10 muestras para análisis
            await self._analyze_trades(symbol)
    
    async def _analyze_orderbook(self, symbol: str) -> None:
        """
        Analizar el OrderBook para un símbolo.
        
        Args:
            symbol: Símbolo a analizar
        """
        # Extraer datos
        order_books = self.order_books[symbol]
        
        # Inicializar resultados para este símbolo si es necesario
        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        
        # Extraer bids y asks
        bids = [ob[1].get("bids", []) for ob in order_books]
        asks = [ob[1].get("asks", []) for ob in order_books]
        
        # Calcular métricas de OrderBook
        bid_volume = sum([sum([bid[1] for bid in book]) for book in bids if book])
        ask_volume = sum([sum([ask[1] for ask in book]) for book in asks if book])
        
        # Calcular desequilibrio de volumen
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            bid_ratio = bid_volume / total_volume
            ask_ratio = ask_volume / total_volume
            imbalance = abs(bid_ratio - ask_ratio)
        else:
            bid_ratio = 0.5
            ask_ratio = 0.5
            imbalance = 0
        
        # Detectar walls (grandes órdenes en un nivel específico)
        bid_walls = []
        ask_walls = []
        
        if bids and bids[-1]:  # Usar el último OrderBook
            last_bids = bids[-1]
            avg_bid_volume = sum([bid[1] for bid in last_bids]) / len(last_bids) if last_bids else 0
            for price, volume in last_bids:
                if volume > avg_bid_volume * 3:  # 3x el volumen promedio
                    bid_walls.append((price, volume))
        
        if asks and asks[-1]:  # Usar el último OrderBook
            last_asks = asks[-1]
            avg_ask_volume = sum([ask[1] for ask in last_asks]) / len(last_asks) if last_asks else 0
            for price, volume in last_asks:
                if volume > avg_ask_volume * 3:  # 3x el volumen promedio
                    ask_walls.append((price, volume))
        
        # Calcular métricas de profundidad
        bid_depth = sum([len(book) for book in bids if book]) / len(bids) if bids else 0
        ask_depth = sum([len(book) for book in asks if book]) / len(asks) if asks else 0
        
        # Guardar resultados
        result = {
            "timestamp": datetime.now().isoformat(),
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "bid_ratio": bid_ratio,
            "ask_ratio": ask_ratio,
            "imbalance": imbalance,
            "bid_walls": bid_walls,
            "ask_walls": ask_walls,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth
        }
        
        # Actualizar resultados
        self.analysis_results[symbol]["orderbook"] = result
        
        # Emitir evento de análisis
        significant_imbalance = imbalance > self.imbalance_threshold
        if significant_imbalance:
            direction = "compra" if bid_ratio > ask_ratio else "venta"
            await self.emit_event("analysis.orderflow_imbalance", {
                "symbol": symbol,
                "imbalance": imbalance,
                "direction": direction,
                "bid_ratio": bid_ratio,
                "ask_ratio": ask_ratio,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Desequilibrio de OrderFlow detectado en {symbol}: {imbalance:.2f} hacia {direction}")
    
    async def _analyze_trades(self, symbol: str) -> None:
        """
        Analizar trades para un símbolo.
        
        Args:
            symbol: Símbolo a analizar
        """
        # Extraer datos
        trade_data = self.trades[symbol]
        
        # Inicializar resultados para este símbolo si es necesario
        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        
        # Extraer trades
        trades = [t[1] for t in trade_data]
        
        # Separar por tipo
        buy_trades = [t for t in trades if t.get("side") == "buy"]
        sell_trades = [t for t in trades if t.get("side") == "sell"]
        
        # Calcular métricas de volumen
        buy_volume = sum([t.get("amount", 0) for t in buy_trades])
        sell_volume = sum([t.get("amount", 0) for t in sell_trades])
        
        # Calcular volumen total y promedio
        total_volume = buy_volume + sell_volume
        avg_trade_size = total_volume / len(trades) if trades else 0
        
        # Identificar trades grandes
        large_trades = [t for t in trades if t.get("amount", 0) > avg_trade_size * 2]
        
        # Calcular desequilibrio de volumen
        if total_volume > 0:
            buy_ratio = buy_volume / total_volume
            sell_ratio = sell_volume / total_volume
            imbalance = abs(buy_ratio - sell_ratio)
        else:
            buy_ratio = 0.5
            sell_ratio = 0.5
            imbalance = 0
        
        # Guardar resultados
        result = {
            "timestamp": datetime.now().isoformat(),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "imbalance": imbalance,
            "avg_trade_size": avg_trade_size,
            "large_trades": len(large_trades)
        }
        
        # Actualizar resultados
        self.analysis_results[symbol]["trades"] = result
        
        # Emitir evento de análisis
        significant_imbalance = imbalance > self.imbalance_threshold
        if significant_imbalance:
            direction = "compra" if buy_ratio > sell_ratio else "venta"
            await self.emit_event("analysis.trade_imbalance", {
                "symbol": symbol,
                "imbalance": imbalance,
                "direction": direction,
                "buy_ratio": buy_ratio,
                "sell_ratio": sell_ratio,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Desequilibrio de trades detectado en {symbol}: {imbalance:.2f} hacia {direction}")
    
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
        
        # Combinar resultados de OrderBook y trades
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # Añadir datos de OrderBook si existen
        if "orderbook" in self.analysis_results[symbol]:
            ob_data = self.analysis_results[symbol]["orderbook"]
            result.update({
                "orderbook_imbalance": ob_data["imbalance"],
                "bid_walls": len(ob_data["bid_walls"]),
                "ask_walls": len(ob_data["ask_walls"]),
                "bid_depth": ob_data["bid_depth"],
                "ask_depth": ob_data["ask_depth"]
            })
        
        # Añadir datos de trades si existen
        if "trades" in self.analysis_results[symbol]:
            trade_data = self.analysis_results[symbol]["trades"]
            result.update({
                "trade_imbalance": trade_data["imbalance"],
                "buy_ratio": trade_data["buy_ratio"],
                "sell_ratio": trade_data["sell_ratio"],
                "avg_trade_size": trade_data["avg_trade_size"],
                "large_trades": trade_data["large_trades"]
            })
        
        # Calcular desequilibrio combinado
        ob_imbalance = self.analysis_results[symbol].get("orderbook", {}).get("imbalance", 0)
        trade_imbalance = self.analysis_results[symbol].get("trades", {}).get("imbalance", 0)
        
        # Ponderación: 60% OrderBook, 40% trades
        combined_imbalance = ob_imbalance * 0.6 + trade_imbalance * 0.4
        result["combined_imbalance"] = combined_imbalance
        
        # Determinar dirección dominante
        ob_buy_bias = self.analysis_results[symbol].get("orderbook", {}).get("bid_ratio", 0.5) > 0.5
        trade_buy_bias = self.analysis_results[symbol].get("trades", {}).get("buy_ratio", 0.5) > 0.5
        
        # Si ambos apuntan en la misma dirección, esa es la dominante
        if ob_buy_bias == trade_buy_bias:
            result["dominant_direction"] = "compra" if ob_buy_bias else "venta"
            result["direction_confidence"] = combined_imbalance
        else:
            # Si no coinciden, usar el que tiene mayor desequilibrio
            if ob_imbalance > trade_imbalance:
                result["dominant_direction"] = "compra" if ob_buy_bias else "venta"
                result["direction_confidence"] = ob_imbalance
            else:
                result["dominant_direction"] = "compra" if trade_buy_bias else "venta"
                result["direction_confidence"] = trade_imbalance
        
        return result
    
    async def generate_report(self, symbol: str) -> Dict[str, Any]:
        """
        Generar reporte completo de OrderFlow para un símbolo.
        
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
        
        # Análisis de desequilibrio
        if analysis.get("combined_imbalance", 0) > self.imbalance_threshold:
            direction = analysis.get("dominant_direction", "")
            insights.append(f"Fuerte desequilibrio ({analysis['combined_imbalance']:.2f}) detectado hacia {direction}")
        
        # Análisis de paredes (walls)
        bid_walls = analysis.get("bid_walls", 0)
        ask_walls = analysis.get("ask_walls", 0)
        if bid_walls > 0:
            insights.append(f"Detectadas {bid_walls} paredes de compra significativas")
        if ask_walls > 0:
            insights.append(f"Detectadas {ask_walls} paredes de venta significativas")
        
        # Análisis de trades grandes
        large_trades = analysis.get("large_trades", 0)
        if large_trades > 0:
            insights.append(f"Detectados {large_trades} trades de gran volumen")
        
        # Añadir al reporte
        analysis["insights"] = insights
        analysis["chart_path"] = chart_path
        
        return analysis
    
    async def _generate_chart(self, symbol: str) -> str:
        """
        Generar gráfico de análisis de OrderFlow.
        
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
            plt.figure(figsize=(12, 8))
            
            # Datos de OrderBook
            if "orderbook" in self.analysis_results[symbol]:
                ob_data = self.analysis_results[symbol]["orderbook"]
                
                # Subplot para desequilibrio de OrderBook
                plt.subplot(2, 2, 1)
                plt.bar(["Compra", "Venta"], [ob_data["bid_ratio"], ob_data["ask_ratio"]])
                plt.title(f"Desequilibrio de OrderBook: {ob_data['imbalance']:.2f}")
                plt.ylim(0, 1)
                
                # Subplot para profundidad
                plt.subplot(2, 2, 2)
                plt.bar(["Compra", "Venta"], [ob_data["bid_depth"], ob_data["ask_depth"]])
                plt.title("Profundidad del Libro")
            
            # Datos de trades
            if "trades" in self.analysis_results[symbol]:
                trade_data = self.analysis_results[symbol]["trades"]
                
                # Subplot para desequilibrio de trades
                plt.subplot(2, 2, 3)
                plt.bar(["Compra", "Venta"], [trade_data["buy_ratio"], trade_data["sell_ratio"]])
                plt.title(f"Desequilibrio de Trades: {trade_data['imbalance']:.2f}")
                plt.ylim(0, 1)
                
                # Subplot para tamaño de trades
                plt.subplot(2, 2, 4)
                plt.bar(["Promedio", "Grandes"], [trade_data["avg_trade_size"], trade_data["large_trades"]])
                plt.title("Tamaño de Trades")
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.plot_dir}/{symbol}_orderflow_{timestamp}.png"
            plt.savefig(file_path)
            plt.close()
            
            return file_path
        
        except Exception as e:
            self.logger.error(f"Error al generar gráfico: {e}")
            return ""


# Función para transformar datos de OrderBook en DataFrame
def orderbook_to_dataframe(bids: List[List[float]], asks: List[List[float]]) -> pd.DataFrame:
    """
    Convertir datos de OrderBook a DataFrame.
    
    Args:
        bids: Lista de órdenes de compra [precio, volumen]
        asks: Lista de órdenes de venta [precio, volumen]
        
    Returns:
        DataFrame con los datos de OrderBook
    """
    # Crear DataFrames separados
    bids_df = pd.DataFrame(bids, columns=["price", "volume"])
    asks_df = pd.DataFrame(asks, columns=["price", "volume"])
    
    # Añadir tipo
    bids_df["type"] = "bid"
    asks_df["type"] = "ask"
    
    # Combinar y ordenar por precio
    df = pd.concat([bids_df, asks_df])
    df = df.sort_values("price")
    
    return df


# Función para detectar absorción en el OrderBook
def detect_absorption(
    orderbook_df: pd.DataFrame, 
    trades: List[Dict[str, Any]],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Detectar absorción en el OrderBook.
    
    La absorción ocurre cuando grandes órdenes son ejecutadas sin
    afectar significativamente el precio.
    
    Args:
        orderbook_df: DataFrame con datos de OrderBook
        trades: Lista de trades recientes
        threshold: Umbral para considerar absorción significativa
        
    Returns:
        Resultado del análisis de absorción
    """
    # Calcular volumen total de trades
    trade_volume = sum([t.get("amount", 0) for t in trades])
    
    # Calcular volumen en el OrderBook
    bid_volume = orderbook_df[orderbook_df["type"] == "bid"]["volume"].sum()
    ask_volume = orderbook_df[orderbook_df["type"] == "ask"]["volume"].sum()
    
    # Calcular ratios
    bid_ratio = trade_volume / bid_volume if bid_volume > 0 else 0
    ask_ratio = trade_volume / ask_volume if ask_volume > 0 else 0
    
    # Determinar si hay absorción
    is_bid_absorption = bid_ratio > threshold
    is_ask_absorption = ask_ratio > threshold
    
    result = {
        "bid_absorption": is_bid_absorption,
        "ask_absorption": is_ask_absorption,
        "bid_ratio": bid_ratio,
        "ask_ratio": ask_ratio,
        "trade_volume": trade_volume,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume
    }
    
    return result


# Exportación para uso fácil
order_flow_analyzer = OrderFlowAnalyzer()