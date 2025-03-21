"""
Motor de backtesting para el sistema Genesis.

Este módulo proporciona un motor avanzado de backtesting para probar
estrategias de trading con datos históricos, incluyendo optimización
de parámetros y análisis de resultados.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Callable, Optional, Tuple, Any, Union
from itertools import product
import os
import asyncio

from genesis.core.base import Component
from genesis.data.market_data import MarketDataManager
from genesis.strategies.base import Strategy, SignalType
from genesis.risk.position_sizer import PositionSizer

class BacktestEngine(Component):
    """
    Motor avanzado de backtesting para estrategias de trading.
    
    Este componente permite realizar backtesting de estrategias de trading
    con datos históricos, incluyendo simulación de comisiones, slippage,
    y análisis de resultados.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        name: str = "backtest_engine"
    ):
        """
        Inicializar el motor de backtesting.
        
        Args:
            initial_capital: Capital inicial para el backtest
            commission: Comisión por operación (fracción del monto total)
            slippage: Deslizamiento por operación (fracción del precio)
            name: Nombre del componente
        """
        super().__init__(name)
        self.initial_capital = max(1000.0, initial_capital)
        self.commission = max(0.0, min(0.01, commission))  # Entre 0% y 1%
        self.slippage = max(0.0, min(0.01, slippage))  # Entre 0% y 1%
        self.market_data = None
        self.position_sizer = PositionSizer()
        self.results = {}
        self.logger = logging.getLogger(__name__)
        self.plots_dir = os.path.join("data", "plots", "backtests")
        os.makedirs(self.plots_dir, exist_ok=True)
        
    async def start(self) -> None:
        """Iniciar el motor de backtesting."""
        await super().start()
        self.logger.info("Motor de backtesting iniciado")
        
    async def stop(self) -> None:
        """Detener el motor de backtesting."""
        await super().stop()
        self.logger.info("Motor de backtesting detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        if event_type == "backtest.request":
            strategy_name = data.get("strategy_name")
            symbol = data.get("symbol")
            timeframe = data.get("timeframe", "1d")
            start_date = data.get("start_date")
            end_date = data.get("end_date")
            params = data.get("params", {})
            
            if not all([strategy_name, symbol, start_date, end_date]):
                self.logger.error("Faltan parámetros obligatorios para el backtest")
                return
                
            try:
                result = await self.run_backtest(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    params=params
                )
                
                await self.emit_event("backtest.result", {
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "result": result,
                    "request_id": data.get("request_id")
                })
                
            except Exception as e:
                self.logger.error(f"Error en backtest: {e}")
                await self.emit_event("backtest.error", {
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "error": str(e),
                    "request_id": data.get("request_id")
                })
                
    def set_market_data(self, market_data: MarketDataManager) -> None:
        """
        Establecer la fuente de datos de mercado.
        
        Args:
            market_data: Instancia de MarketDataManager
        """
        self.market_data = market_data
        self.logger.info("Fuente de datos de mercado establecida")
        
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        exchange: str = "binance"
    ) -> pd.DataFrame:
        """
        Obtener datos históricos para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            timeframe: Intervalo de tiempo (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_date: Fecha de inicio (formato YYYY-MM-DD)
            end_date: Fecha de fin (formato YYYY-MM-DD)
            exchange: Nombre del exchange
            
        Returns:
            DataFrame con datos históricos
        """
        if self.market_data is None:
            raise ValueError("No se ha establecido la fuente de datos de mercado")
            
        try:
            self.logger.info(f"Obteniendo datos para {symbol} en {timeframe}")
            
            # Este método debe implementarse en MarketDataManager
            # para proporcionar datos históricos
            data = await self.market_data.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                exchange=exchange
            )
            
            if data.empty:
                self.logger.warning(f"No se encontraron datos para {symbol}")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error al obtener datos históricos: {e}")
            return pd.DataFrame()
            
    async def calculate_indicators(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calcular indicadores técnicos para el backtest.
        
        Args:
            df: DataFrame con datos históricos
            params: Parámetros para los indicadores
            
        Returns:
            DataFrame con indicadores calculados
        """
        # Verificar que el DataFrame tenga los campos necesarios
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns and col.capitalize() not in df.columns:
                self.logger.error(f"Columna '{col}' no encontrada en los datos")
                return df
                
        # Normalizar nombres de columnas
        for col in required_columns:
            if col.capitalize() in df.columns and col not in df.columns:
                df[col] = df[col.capitalize()]
                
        try:
            # Calcular indicadores básicos
            # En una implementación real, esto podría usar talib u otra biblioteca
            # Aquí implementamos algunos indicadores básicos manualmente
            
            # Media móvil simple
            fast_period = params.get("sma_fast_period", 20)
            slow_period = params.get("sma_slow_period", 50)
            df["sma_fast"] = df["close"].rolling(window=fast_period).mean()
            df["sma_slow"] = df["close"].rolling(window=slow_period).mean()
            
            # RSI
            rsi_period = params.get("rsi_period", 14)
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # MACD
            fast_period = params.get("macd_fast_period", 12)
            slow_period = params.get("macd_slow_period", 26)
            signal_period = params.get("macd_signal_period", 9)
            
            ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"Error al calcular indicadores: {e}")
            return df
            
    async def run_strategy(
        self,
        strategy: Union[Strategy, Callable],
        df: pd.DataFrame,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Ejecutar una estrategia de trading en datos históricos.
        
        Args:
            strategy: Estrategia de trading (instancia de Strategy o función)
            df: DataFrame con datos históricos e indicadores
            params: Parámetros para la estrategia
            
        Returns:
            DataFrame con señales de trading
        """
        try:
            # Si es una función de estrategia
            if callable(strategy) and not isinstance(strategy, Strategy):
                return strategy(df, params)
                
            # Si es una instancia de Strategy
            if isinstance(strategy, Strategy):
                signals = []
                for idx, row in df.iterrows():
                    # Convertir la fila a un formato que la estrategia pueda usar
                    data_point = row.to_dict()
                    signal = await strategy.generate_signal(data_point.get("symbol", ""), pd.DataFrame([data_point]))
                    signals.append(signal.get("signal", SignalType.HOLD))
                    
                df["signal"] = signals
                return df
                
            raise ValueError("La estrategia proporcionada no es válida")
            
        except Exception as e:
            self.logger.error(f"Error al ejecutar estrategia: {e}")
            df["signal"] = SignalType.HOLD
            return df
            
    async def simulate_trading(
        self,
        df: pd.DataFrame,
        initial_capital: float = None
    ) -> Dict[str, Any]:
        """
        Simular operaciones de trading basadas en señales.
        
        Args:
            df: DataFrame con señales de trading
            initial_capital: Capital inicial para la simulación
            
        Returns:
            Resultados de la simulación
        """
        if initial_capital is None:
            initial_capital = self.initial_capital
            
        if "signal" not in df.columns:
            self.logger.error("No se encontraron señales en los datos")
            return {}
            
        try:
            # Inicializar variables para la simulación
            capital = initial_capital
            position = 0
            equity = []
            trades = []
            position_value = []
            cash = []
            
            for idx, row in df.iterrows():
                signal = str(row["signal"]).lower()
                close_price = row["close"]
                
                # Aplicar slippage
                execution_price = close_price * (1 + self.slippage if signal == SignalType.BUY.lower() else 
                                              1 - self.slippage if signal == SignalType.SELL.lower() else 1)
                
                # Procesar señal
                if signal == SignalType.BUY.lower() and position == 0:  # Compra
                    position = capital * (1 - self.commission) / execution_price
                    capital = 0
                    trades.append({
                        "type": "buy",
                        "price": execution_price,
                        "time": idx,
                        "commission": position * execution_price * self.commission
                    })
                    
                elif signal == SignalType.SELL.lower() and position > 0:  # Venta
                    capital = position * execution_price * (1 - self.commission)
                    commission = position * execution_price * self.commission
                    trades.append({
                        "type": "sell",
                        "price": execution_price,
                        "time": idx,
                        "commission": commission
                    })
                    position = 0
                    
                elif signal == SignalType.EXIT.lower() and position > 0:  # Cerrar posición
                    capital = position * execution_price * (1 - self.commission)
                    commission = position * execution_price * self.commission
                    trades.append({
                        "type": "exit",
                        "price": execution_price,
                        "time": idx,
                        "commission": commission
                    })
                    position = 0
                    
                # Calcular equity en cada punto
                pos_value = position * close_price
                equity.append(capital + pos_value)
                position_value.append(pos_value)
                cash.append(capital)
                
            # Añadir resultados al DataFrame
            df["equity"] = equity
            df["position_value"] = position_value
            df["cash"] = cash
            
            return self.calculate_metrics(df, trades)
            
        except Exception as e:
            self.logger.error(f"Error en simulación de trading: {e}")
            return {}
            
    def calculate_metrics(self, df: pd.DataFrame, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcular métricas de rendimiento del backtest.
        
        Args:
            df: DataFrame con resultados de la simulación
            trades: Lista de operaciones realizadas
            
        Returns:
            Métricas de rendimiento
        """
        try:
            # Calcular retornos
            returns = df["equity"].pct_change().fillna(0)
            
            # Retorno total
            total_return = (df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
            
            # Sharpe Ratio (anualizado)
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
            
            # Maximum Drawdown
            rolling_max = df["equity"].cummax()
            drawdown = (rolling_max - df["equity"]) / rolling_max
            max_drawdown = drawdown.max()
            
            # Calcular ganancias/pérdidas por operación
            if len(trades) > 0:
                buy_trades = [t for t in trades if t["type"] == "buy"]
                sell_trades = [t for t in trades if t["type"] in ("sell", "exit")]
                
                if len(buy_trades) == len(sell_trades):
                    pnl = []
                    for i in range(len(buy_trades)):
                        buy_price = buy_trades[i]["price"]
                        sell_price = sell_trades[i]["price"]
                        buy_commission = buy_trades[i]["commission"]
                        sell_commission = sell_trades[i]["commission"]
                        pnl.append({
                            "entry_time": buy_trades[i]["time"],
                            "exit_time": sell_trades[i]["time"],
                            "entry_price": buy_price,
                            "exit_price": sell_price,
                            "pnl_pct": (sell_price - buy_price) / buy_price,
                            "commission": buy_commission + sell_commission
                        })
                else:
                    pnl = []
            else:
                pnl = []
                
            # Calcular win rate
            win_trades = sum(1 for p in pnl if p["pnl_pct"] > 0)
            total_trades = len(pnl)
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "num_trades": len(trades) // 2,  # Cada operación completa = entrada + salida
                "win_rate": win_rate,
                "equity_curve": df["equity"].tolist(),
                "trades": trades,
                "pnl": pnl,
                "first_date": df.index[0],
                "last_date": df.index[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error al calcular métricas: {e}")
            return {}
            
    async def optimize_strategy(
        self,
        strategy: Union[Strategy, Callable],
        df: pd.DataFrame,
        param_grid: Dict[str, List],
        metric: str = "sharpe_ratio"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimizar parámetros de una estrategia.
        
        Args:
            strategy: Estrategia de trading
            df: DataFrame con datos históricos
            param_grid: Grid de parámetros a probar
            metric: Métrica a optimizar
            
        Returns:
            Tuple de (mejor resultado, mejores parámetros)
        """
        best_result = None
        best_params = None
        best_metric_value = float("-inf")
        
        # Generar todas las combinaciones de parámetros
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        # Ejecutar backtest para cada combinación
        for params in param_combinations:
            self.logger.debug(f"Probando parámetros: {params}")
            
            # Calcular indicadores
            df_with_indicators = await self.calculate_indicators(df.copy(), params)
            
            # Ejecutar estrategia
            df_with_signals = await self.run_strategy(strategy, df_with_indicators, params)
            
            # Simular trading
            result = await self.simulate_trading(df_with_signals)
            
            # Evaluar métrica
            if metric in result and (best_result is None or result[metric] > best_metric_value):
                best_metric_value = result[metric]
                best_result = result
                best_params = params
                
        if best_result is None:
            raise ValueError("No se pudo optimizar la estrategia con los parámetros dados")
            
        return best_result, best_params
        
    async def run_backtest(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        params: Dict[str, Any] = None,
        exchange: str = "binance"
    ) -> Dict[str, Any]:
        """
        Ejecutar un backtest completo.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo de trading
            timeframe: Intervalo de tiempo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            params: Parámetros para la estrategia
            exchange: Nombre del exchange
            
        Returns:
            Resultados del backtest
        """
        try:
            if params is None:
                params = {}
                
            # Obtener datos históricos
            df = await self.fetch_historical_data(symbol, timeframe, start_date, end_date, exchange)
            
            if df.empty:
                raise ValueError(f"No se encontraron datos para {symbol}")
                
            # Obtener estrategia
            strategy = await self.get_strategy(strategy_name)
            
            if strategy is None:
                raise ValueError(f"Estrategia '{strategy_name}' no encontrada")
                
            # Calcular indicadores
            df_with_indicators = await self.calculate_indicators(df, params)
            
            # Ejecutar estrategia
            df_with_signals = await self.run_strategy(strategy, df_with_indicators, params)
            
            # Simular trading
            result = await self.simulate_trading(df_with_signals)
            
            # Guardar resultado
            self.results[f"{strategy_name}_{symbol}"] = {
                "metrics": result,
                "params": params,
                "df": df_with_signals
            }
            
            self.logger.info(f"Backtest completado para {strategy_name} en {symbol}")
            self.logger.info(f"Retorno: {result.get('total_return', 0):.2%}, Sharpe: {result.get('sharpe_ratio', 0):.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al ejecutar backtest: {e}")
            raise
            
    async def get_strategy(self, strategy_name: str) -> Union[Strategy, Callable, None]:
        """
        Obtener una estrategia por nombre.
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            Estrategia de trading o None si no se encuentra
        """
        # En una implementación real, esto buscaría en un registro de estrategias
        # o cargaría dinámicamente una estrategia desde un módulo
        
        # Por ahora implementamos algunas estrategias de ejemplo
        if strategy_name == "trend_following":
            return self.trend_following_strategy
        elif strategy_name == "mean_reversion":
            return self.mean_reversion_strategy
        else:
            # Buscar en el bus de eventos o en el sistema
            await self.emit_event("strategy.request", {"strategy_name": strategy_name})
            # Por ahora retornamos None, pero en una implementación real
            # esperaríamos a que el componente de estrategia responda
            return None
            
    def trend_following_strategy(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Estrategia de seguimiento de tendencias basada en SMA.
        
        Args:
            df: DataFrame con indicadores
            params: Parámetros de la estrategia
            
        Returns:
            DataFrame con señales
        """
        df["signal"] = SignalType.HOLD.lower()
        
        # Verificar que tenemos los indicadores necesarios
        if "sma_fast" not in df.columns or "sma_slow" not in df.columns:
            return df
            
        # Generar señales
        df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = SignalType.BUY.lower()
        df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = SignalType.SELL.lower()
        
        return df
        
    def mean_reversion_strategy(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Estrategia de reversión a la media basada en RSI.
        
        Args:
            df: DataFrame con indicadores
            params: Parámetros de la estrategia
            
        Returns:
            DataFrame con señales
        """
        df["signal"] = SignalType.HOLD.lower()
        
        # Verificar que tenemos los indicadores necesarios
        if "rsi" not in df.columns:
            return df
            
        # Obtener parámetros
        rsi_low = params.get("rsi_low", 30)
        rsi_high = params.get("rsi_high", 70)
        
        # Generar señales
        df.loc[df["rsi"] < rsi_low, "signal"] = SignalType.BUY.lower()
        df.loc[df["rsi"] > rsi_high, "signal"] = SignalType.SELL.lower()
        
        return df
        
    def plot_results(self, strategy_name: str, symbol: str, show: bool = True, save: bool = False) -> Optional[str]:
        """
        Generar gráficos de resultados de backtest.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo de trading
            show: Si se debe mostrar el gráfico
            save: Si se debe guardar el gráfico
            
        Returns:
            Ruta al archivo guardado o None
        """
        key = f"{strategy_name}_{symbol}"
        if key not in self.results:
            self.logger.error(f"No hay resultados para {key}")
            return None
            
        try:
            result = self.results[key]
            metrics = result["metrics"]
            df = result["df"]
            
            # Crear figura
            plt.figure(figsize=(12, 8))
            
            # Graficar precio y equity
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_title(f"Backtest: {strategy_name} en {symbol}")
            ax1.plot(df.index, df["close"], label="Precio", color="blue", alpha=0.6)
            
            # Marcar operaciones
            buy_signals = df[df["signal"] == SignalType.BUY.lower()]
            sell_signals = df[df["signal"] == SignalType.SELL.lower()]
            
            ax1.scatter(buy_signals.index, buy_signals["close"], marker="^", color="green", 
                      label="Compra", alpha=1, s=50)
            ax1.scatter(sell_signals.index, sell_signals["close"], marker="v", color="red", 
                      label="Venta", alpha=1, s=50)
            
            ax1.set_ylabel("Precio")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Graficar equity
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(df.index, df["equity"], label="Equity", color="purple")
            ax2.set_ylabel("Equity")
            ax2.set_xlabel("Fecha")
            ax2.grid(True, alpha=0.3)
            
            # Añadir métricas como texto
            metrics_text = (
                f"Retorno: {metrics.get('total_return', 0):.2%}  "
                f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  "
                f"Drawdown: {metrics.get('max_drawdown', 0):.2%}  "
                f"Win Rate: {metrics.get('win_rate', 0):.2%}  "
                f"Ops: {metrics.get('num_trades', 0)}"
            )
            plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=10, 
                       bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)
            
            # Guardar o mostrar
            saved_path = None
            if save:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{key}_{timestamp}.png"
                filepath = os.path.join(self.plots_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                saved_path = filepath
                self.logger.info(f"Gráfico guardado en {filepath}")
                
            if show:
                plt.show()
            else:
                plt.close()
                
            return saved_path
            
        except Exception as e:
            self.logger.error(f"Error al generar gráfico: {e}")
            return None
            
    def get_result(self, strategy_name: str, symbol: str) -> Dict[str, Any]:
        """
        Obtener los resultados de un backtest.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo de trading
            
        Returns:
            Resultados del backtest o diccionario vacío si no hay resultados
        """
        key = f"{strategy_name}_{symbol}"
        return self.results.get(key, {})