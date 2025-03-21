"""
Motor de backtesting para el sistema Genesis.

Este módulo proporciona un motor avanzado para realizar backtesting de
estrategias de trading utilizando datos históricos de mercado, simulando
ejecuciones, slippage, y otras condiciones de mercado reales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import os
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type
import logging
import uuid
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from genesis.core.base import Component
from genesis.utils.logger import setup_logging
from genesis.strategies.base import Strategy
from genesis.risk.position_sizer import PositionSizer
from genesis.risk.stop_loss import StopLossCalculator


class BacktestResult:
    """
    Resultado de un backtest.
    
    Esta clase encapsula los resultados completos de un backtest,
    incluyendo trades, métricas, y gráficos.
    """
    
    def __init__(self, 
                strategy_name: str,
                symbol: str,
                timeframe: str,
                start_date: datetime,
                end_date: datetime,
                initial_balance: float):
        """
        Inicializar el resultado de backtest.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo de trading
            timeframe: Timeframe utilizado
            start_date: Fecha de inicio
            end_date: Fecha de fin
            initial_balance: Balance inicial
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        
        # Trades y balance
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = [initial_balance]
        self.drawdowns: List[float] = [0.0]
        
        # Métricas
        self.metrics: Dict[str, Any] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_profit_per_trade": 0.0,
            "avg_loss_per_trade": 0.0,
            "expectancy": 0.0,
            "annual_return": 0.0,
            "recovery_factor": 0.0
        }
        
        # Configuración de la estrategia
        self.strategy_config: Dict[str, Any] = {}
        
        # Rutas a los gráficos
        self.charts: Dict[str, str] = {}
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Añadir un trade al resultado.
        
        Args:
            trade: Datos del trade
        """
        self.trades.append(trade)
        
        # Actualizar equity curve
        current_balance = self.equity_curve[-1] + trade["profit"]
        self.equity_curve.append(current_balance)
        
        # Calcular drawdown
        peak = max(self.equity_curve)
        drawdown = peak - current_balance
        drawdown_pct = drawdown / peak if peak > 0 else 0
        self.drawdowns.append(drawdown_pct)
    
    def calculate_metrics(self) -> None:
        """Calcular métricas a partir de los trades."""
        if not self.trades:
            return
        
        # Métricas básicas
        self.metrics["total_trades"] = len(self.trades)
        
        # Separar trades ganadores y perdedores
        winning_trades = [t for t in self.trades if t["profit"] > 0]
        losing_trades = [t for t in self.trades if t["profit"] <= 0]
        
        self.metrics["winning_trades"] = len(winning_trades)
        self.metrics["losing_trades"] = len(losing_trades)
        
        # Win rate
        self.metrics["win_rate"] = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Profit/Loss
        self.metrics["profit_loss"] = sum(t["profit"] for t in self.trades)
        
        # Profit factor
        total_profit = sum(t["profit"] for t in winning_trades)
        total_loss = abs(sum(t["profit"] for t in losing_trades))
        self.metrics["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Avg profit/loss per trade
        self.metrics["avg_profit_per_trade"] = total_profit / len(winning_trades) if winning_trades else 0
        self.metrics["avg_loss_per_trade"] = total_loss / len(losing_trades) if losing_trades else 0
        
        # Expectancy
        self.metrics["expectancy"] = (self.metrics["win_rate"] * self.metrics["avg_profit_per_trade"]) - \
                                    ((1 - self.metrics["win_rate"]) * self.metrics["avg_loss_per_trade"])
        
        # Max drawdown
        max_drawdown_pct = max(self.drawdowns) if self.drawdowns else 0
        self.metrics["max_drawdown_pct"] = max_drawdown_pct
        
        # Sharpe y Sortino
        returns = []
        for i in range(1, len(self.equity_curve)):
            returns.append((self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1])
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe ratio (asumiendo tasa libre de riesgo = 0)
            self.metrics["sharpe_ratio"] = avg_return / std_return if std_return > 0 else 0
            
            # Sortino ratio (solo considerando retornos negativos)
            neg_returns = [r for r in returns if r < 0]
            std_neg_return = np.std(neg_returns) if neg_returns else 0
            self.metrics["sortino_ratio"] = avg_return / std_neg_return if std_neg_return > 0 else 0
        
        # Annual return
        days = (self.end_date - self.start_date).days
        if days > 0 and self.initial_balance > 0:
            final_balance = self.equity_curve[-1]
            total_return = (final_balance - self.initial_balance) / self.initial_balance
            annual_return = ((1 + total_return) ** (365 / days)) - 1
            self.metrics["annual_return"] = annual_return
        
        # Recovery factor
        max_drawdown_amount = max_drawdown_pct * max(self.equity_curve)
        self.metrics["max_drawdown"] = max_drawdown_amount
        self.metrics["recovery_factor"] = self.metrics["profit_loss"] / max_drawdown_amount if max_drawdown_amount > 0 else float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir el resultado a diccionario.
        
        Returns:
            Diccionario con los resultados
        """
        result = {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_balance": self.initial_balance,
            "final_balance": self.equity_curve[-1] if self.equity_curve else self.initial_balance,
            "trades_count": len(self.trades),
            "metrics": self.metrics,
            "strategy_config": self.strategy_config,
            "charts": self.charts
        }
        
        return result
    
    def save(self, file_path: str) -> bool:
        """
        Guardar el resultado a un archivo JSON.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Convertir a diccionario
            data = self.to_dict()
            
            # Convertir fechas a strings
            data["start_date"] = self.start_date.isoformat()
            data["end_date"] = self.end_date.isoformat()
            
            # Guardar
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error al guardar resultados: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> 'BacktestResult':
        """
        Cargar resultados desde un archivo JSON.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Instancia de BacktestResult
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Crear instancia
            start_date = datetime.fromisoformat(data["start_date"])
            end_date = datetime.fromisoformat(data["end_date"])
            
            result = cls(
                data["strategy_name"],
                data["symbol"],
                data["timeframe"],
                start_date,
                end_date,
                data["initial_balance"]
            )
            
            # Restaurar datos
            result.metrics = data["metrics"]
            result.strategy_config = data["strategy_config"]
            result.charts = data["charts"]
            
            # Recrear equity curve
            result.equity_curve = [data["initial_balance"], data["final_balance"]]
            
            return result
        except Exception as e:
            print(f"Error al cargar resultados: {e}")
            return None


class BacktestEngine(Component):
    """
    Motor de backtesting para estrategias de trading.
    
    Este componente permite realizar backtesting de estrategias utilizando
    datos históricos, con simulación de condiciones de mercado reales.
    """
    
    def __init__(self, name: str = "backtest_engine"):
        """
        Inicializar el motor de backtesting.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Configuración
        self.data_dir = "data/historical"
        self.results_dir = "data/backtest_results"
        self.plots_dir = f"{self.results_dir}/plots"
        
        # Crear directorios si no existen
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Position sizing y stop loss
        self.position_sizer = PositionSizer()
        self.stop_loss_calculator = StopLossCalculator()
    
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
        # Procesar eventos relacionados con backtesting
        if event_type == "backtest.run":
            strategy_name = data.get("strategy_name")
            strategy_class = data.get("strategy_class")
            strategy_params = data.get("strategy_params", {})
            symbol = data.get("symbol")
            timeframe = data.get("timeframe", "1h")
            start_date = data.get("start_date")
            end_date = data.get("end_date")
            initial_balance = data.get("initial_balance", 10000.0)
            
            # Ejecutar backtest
            result = await self.run_backtest(
                strategy_class,
                strategy_params,
                symbol,
                timeframe,
                start_date,
                end_date,
                initial_balance
            )
            
            # Emitir evento con resultados
            if result:
                await self.emit_event("backtest.completed", {
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "metrics": result.metrics,
                    "result_id": str(uuid.uuid4())
                })
    
    async def run_backtest(
        self, 
        strategy_class: Type[Strategy],
        strategy_params: Dict[str, Any],
        symbol: str,
        timeframe: str = "1h",
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        initial_balance: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
        risk_per_trade: float = 0.02,  # 2%
        generate_plots: bool = True
    ) -> Optional[BacktestResult]:
        """
        Ejecutar un backtest para una estrategia.
        
        Args:
            strategy_class: Clase de la estrategia
            strategy_params: Parámetros de la estrategia
            symbol: Símbolo de trading
            timeframe: Timeframe para los datos
            start_date: Fecha de inicio
            end_date: Fecha de fin
            initial_balance: Balance inicial
            commission: Comisión por operación (porcentaje)
            slippage: Slippage por operación (porcentaje)
            risk_per_trade: Riesgo por operación (porcentaje)
            generate_plots: Si se deben generar gráficos
            
        Returns:
            Resultados del backtest
        """
        self.logger.info(f"Iniciando backtest para {strategy_class.__name__} en {symbol} ({timeframe})")
        
        # Convertir fechas si son strings
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Si no se especifican fechas, usar últimos 3 meses
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=90)
        
        # Cargar datos históricos
        data = await self._load_historical_data(symbol, timeframe, start_date, end_date)
        if data is None or len(data) < 10:
            self.logger.error(f"Datos insuficientes para {symbol} ({timeframe})")
            return None
        
        # Inicializar resultado
        result = BacktestResult(
            strategy_class.__name__,
            symbol,
            timeframe,
            start_date,
            end_date,
            initial_balance
        )
        
        # Guardar configuración
        result.strategy_config = {
            "params": strategy_params,
            "commission": commission,
            "slippage": slippage,
            "risk_per_trade": risk_per_trade
        }
        
        # Inicializar estrategia
        strategy = strategy_class(**strategy_params)
        
        # Variables para el backtest
        balance = initial_balance
        position = None
        
        # Recorrer los datos
        for i in range(len(data)):
            # Obtener datos hasta este punto (excluir datos futuros)
            current_data = data.iloc[:i+1].copy()
            
            if i < 50:  # Requerir al menos 50 barras para iniciar
                continue
            
            # Generar señal
            signal = await strategy.generate_signal(symbol, current_data)
            signal_type = signal.get("type", "hold")
            
            # Precio actual
            current_price = current_data.iloc[-1]["close"]
            current_time = current_data.index[-1]
            
            # Procesar señal
            if position is None:  # Sin posición abierta
                if signal_type in ("buy", "sell"):
                    # Calcular tamaño de posición
                    position_size = self.position_sizer.calculate(
                        balance, 
                        risk_per_trade, 
                        signal.get("strength", 1.0)
                    )
                    
                    # Aplicar comisión y slippage
                    entry_price = current_price * (1 + slippage) if signal_type == "buy" else current_price * (1 - slippage)
                    position_cost = position_size * (1 + commission)
                    
                    # Cantidad de unidades
                    units = position_size / entry_price
                    
                    # Crear posición
                    position = {
                        "type": signal_type,
                        "entry_price": entry_price,
                        "units": units,
                        "size": position_size,
                        "entry_time": current_time,
                        "stop_loss": signal.get("stop_loss", None),
                        "take_profit": signal.get("take_profit", None)
                    }
                    
                    # Restar comisión del balance
                    balance -= position_size * commission
                    
                    # Registrar operación
                    self.logger.debug(f"Abriendo posición {signal_type} a {entry_price} ({units} unidades)")
            
            else:  # Con posición abierta
                # Verificar stop loss / take profit
                exit_triggered = False
                exit_reason = ""
                
                if position["type"] == "buy":
                    # Stop loss
                    if position["stop_loss"] and current_price <= position["stop_loss"]:
                        exit_triggered = True
                        exit_reason = "stop_loss"
                    
                    # Take profit
                    elif position["take_profit"] and current_price >= position["take_profit"]:
                        exit_triggered = True
                        exit_reason = "take_profit"
                
                elif position["type"] == "sell":
                    # Stop loss
                    if position["stop_loss"] and current_price >= position["stop_loss"]:
                        exit_triggered = True
                        exit_reason = "stop_loss"
                    
                    # Take profit
                    elif position["take_profit"] and current_price <= position["take_profit"]:
                        exit_triggered = True
                        exit_reason = "take_profit"
                
                # Señal de cierre
                if signal_type in ("exit", "close") or signal_type == "buy" and position["type"] == "sell" or signal_type == "sell" and position["type"] == "buy":
                    exit_triggered = True
                    exit_reason = "signal"
                
                # Si se debe cerrar la posición
                if exit_triggered:
                    # Aplicar slippage al precio de salida
                    exit_price = current_price * (1 - slippage) if position["type"] == "buy" else current_price * (1 + slippage)
                    
                    # Calcular ganancia/pérdida
                    if position["type"] == "buy":
                        profit = (exit_price - position["entry_price"]) * position["units"]
                    else:  # sell
                        profit = (position["entry_price"] - exit_price) * position["units"]
                    
                    # Aplicar comisión
                    profit -= position["size"] * commission
                    
                    # Actualizar balance
                    balance += position["size"] + profit
                    
                    # Registrar trade
                    trade = {
                        "id": str(uuid.uuid4()),
                        "symbol": symbol,
                        "type": position["type"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "entry_time": position["entry_time"].isoformat(),
                        "exit_time": current_time.isoformat(),
                        "units": position["units"],
                        "size": position["size"],
                        "profit": profit,
                        "profit_pct": profit / position["size"],
                        "exit_reason": exit_reason
                    }
                    
                    # Añadir trade al resultado
                    result.add_trade(trade)
                    
                    # Resetear posición
                    position = None
                    
                    self.logger.debug(f"Cerrando posición a {exit_price}, profit: {profit:.2f}")
        
        # Cerrar posición abierta al final del backtest
        if position is not None:
            # Último precio
            last_price = data.iloc[-1]["close"]
            
            # Aplicar slippage
            exit_price = last_price * (1 - slippage) if position["type"] == "buy" else last_price * (1 + slippage)
            
            # Calcular ganancia/pérdida
            if position["type"] == "buy":
                profit = (exit_price - position["entry_price"]) * position["units"]
            else:  # sell
                profit = (position["entry_price"] - exit_price) * position["units"]
            
            # Aplicar comisión
            profit -= position["size"] * commission
            
            # Actualizar balance
            balance += position["size"] + profit
            
            # Registrar trade
            trade = {
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "type": position["type"],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "entry_time": position["entry_time"].isoformat(),
                "exit_time": data.index[-1].isoformat(),
                "units": position["units"],
                "size": position["size"],
                "profit": profit,
                "profit_pct": profit / position["size"],
                "exit_reason": "end_of_test"
            }
            
            # Añadir trade al resultado
            result.add_trade(trade)
            
            self.logger.debug(f"Cerrando posición final a {exit_price}, profit: {profit:.2f}")
        
        # Calcular métricas
        result.calculate_metrics()
        
        # Generar gráficos
        if generate_plots:
            await self._generate_plots(result, data)
        
        # Guardar resultado
        result_path = f"{self.results_dir}/{strategy_class.__name__}_{symbol}_{timeframe}_{int(time.time())}.json"
        result.save(result_path)
        
        self.logger.info(f"Backtest completado para {strategy_class.__name__} en {symbol} ({timeframe})")
        self.logger.info(f"Trades: {result.metrics['total_trades']}, Win Rate: {result.metrics['win_rate']:.2%}, P/L: {result.metrics['profit_loss']:.2f}")
        
        return result
    
    async def _load_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Cargar datos históricos para el backtest.
        
        Args:
            symbol: Símbolo de trading
            timeframe: Timeframe para los datos
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            DataFrame con los datos históricos
        """
        # Intentar cargar desde archivo
        file_path = f"{self.data_dir}/{symbol}_{timeframe}.csv"
        
        if os.path.exists(file_path):
            try:
                # Cargar datos
                data = pd.read_csv(file_path)
                
                # Convertir timestamp a datetime y establecer como índice
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                data.set_index("timestamp", inplace=True)
                
                # Filtrar por fechas
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                return data
            except Exception as e:
                self.logger.error(f"Error al cargar datos históricos: {e}")
        
        # Si no existe el archivo o hubo error, intentar cargar desde la base de datos
        # Este método dependerá de la implementación específica de cómo se accede a los datos históricos
        
        # Si todo falla, generar datos sintéticos para testing
        # NOTA: En una implementación real, esto debería reemplazarse por carga desde API o BD
        self.logger.warning(f"Usando datos sintéticos para {symbol} ({timeframe})")
        
        # Generar fechas
        date_range = pd.date_range(start=start_date, end=end_date, freq=self._timeframe_to_pandas_freq(timeframe))
        
        # Generar precios sintéticos
        np.random.seed(42)  # Para reproducibilidad
        
        # Parámetros base
        base_price = 100.0
        volatility = 0.01
        
        # Generar precios como un paseo aleatorio
        price_changes = np.random.normal(0, volatility, size=len(date_range))
        
        # Convertir a cambios porcentuales
        price_multipliers = 1.0 + price_changes
        
        # Calcular precios acumulativos
        prices = base_price * np.cumprod(price_multipliers)
        
        # Generar OHLCV
        opens = prices.copy()
        closes = np.roll(prices, -1)  # Precio siguiente
        closes[-1] = closes[-2] * (1 + np.random.normal(0, volatility))  # Último precio
        
        # Generar high y low
        high_offsets = np.abs(np.random.normal(0, volatility * 2, size=len(date_range)))
        low_offsets = np.abs(np.random.normal(0, volatility * 2, size=len(date_range)))
        
        highs = np.maximum(opens, closes) + np.maximum(opens, closes) * high_offsets
        lows = np.minimum(opens, closes) - np.minimum(opens, closes) * low_offsets
        
        # Asegurar que high >= max(open, close) y low <= min(open, close)
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # Generar volumen
        volumes = np.random.gamma(2.0, 100000, size=len(date_range))
        
        # Crear DataFrame
        data = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        }, index=date_range)
        
        return data
    
    def _timeframe_to_pandas_freq(self, timeframe: str) -> str:
        """
        Convertir timeframe a frecuencia de pandas.
        
        Args:
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            
        Returns:
            Frecuencia de pandas
        """
        # Extraer número y unidad
        if len(timeframe) < 2:
            return 'D'  # Default diario
        
        num = int(timeframe[:-1])
        unit = timeframe[-1].lower()
        
        # Convertir
        if unit == 'm':
            return f'{num}T'  # Minutos
        elif unit == 'h':
            return f'{num}H'  # Horas
        elif unit == 'd':
            return f'{num}D'  # Días
        elif unit == 'w':
            return f'{num}W'  # Semanas
        elif unit == 'M':
            return f'{num}M'  # Meses
        else:
            return 'D'  # Default diario
    
    async def _generate_plots(self, result: BacktestResult, data: pd.DataFrame) -> None:
        """
        Generar gráficos para los resultados del backtest.
        
        Args:
            result: Resultado del backtest
            data: Datos históricos
        """
        strategy_name = result.strategy_name
        symbol = result.symbol
        timestamp = int(time.time())
        
        try:
            # 1. Equity curve
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(result.equity_curve)), result.equity_curve)
            plt.title(f'Equity Curve - {strategy_name} ({symbol})')
            plt.xlabel('Trades')
            plt.ylabel('Balance')
            plt.grid(True)
            
            equity_path = f"{self.plots_dir}/equity_{strategy_name}_{symbol}_{timestamp}.png"
            plt.savefig(equity_path)
            plt.close()
            
            result.charts["equity_curve"] = equity_path
            
            # 2. Drawdown
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(result.drawdowns)), [d * 100 for d in result.drawdowns])
            plt.title(f'Drawdown - {strategy_name} ({symbol})')
            plt.xlabel('Trades')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            
            drawdown_path = f"{self.plots_dir}/drawdown_{strategy_name}_{symbol}_{timestamp}.png"
            plt.savefig(drawdown_path)
            plt.close()
            
            result.charts["drawdown"] = drawdown_path
            
            # 3. Trades en el gráfico de precios
            if len(result.trades) > 0 and len(data) > 0:
                # Preparar gráfico de precios
                plt.figure(figsize=(12, 8))
                
                # Convertir índice a lista para slicing
                dates = data.index.tolist()
                
                # Plotear gráfico de precios
                plt.plot(dates, data['close'], color='black', alpha=0.6)
                plt.title(f'Trades - {strategy_name} ({symbol})')
                plt.xlabel('Fecha')
                plt.ylabel('Precio')
                
                # Plotear trades
                for trade in result.trades:
                    try:
                        entry_time = datetime.fromisoformat(trade["entry_time"])
                        exit_time = datetime.fromisoformat(trade["exit_time"])
                        
                        # Encontrar índices más cercanos
                        entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
                        exit_idx = data.index.get_indexer([exit_time], method='nearest')[0]
                        
                        # Si los índices son válidos
                        if 0 <= entry_idx < len(data) and 0 <= exit_idx < len(data):
                            # Color según tipo de trade
                            color = 'green' if trade["type"] == "buy" else 'red'
                            
                            # Entrada
                            plt.scatter(dates[entry_idx], trade["entry_price"], 
                                       color=color, marker='^' if trade["type"] == "buy" else 'v', s=100)
                            
                            # Salida
                            plt.scatter(dates[exit_idx], trade["exit_price"], 
                                       color='blue', marker='o', s=100)
                            
                            # Línea conectando entrada y salida
                            plt.plot([dates[entry_idx], dates[exit_idx]], 
                                   [trade["entry_price"], trade["exit_price"]], 
                                   color=color, linestyle='--', alpha=0.7)
                    except (ValueError, KeyError, IndexError) as e:
                        self.logger.error(f"Error al plotear trade: {e}")
                
                plt.grid(True)
                plt.xticks(rotation=45)
                
                trades_path = f"{self.plots_dir}/trades_{strategy_name}_{symbol}_{timestamp}.png"
                plt.savefig(trades_path)
                plt.close()
                
                result.charts["trades"] = trades_path
            
            # 4. Distribución de ganancias/pérdidas
            plt.figure(figsize=(10, 6))
            profits = [t["profit"] for t in result.trades]
            plt.hist(profits, bins=20, alpha=0.7, color='blue')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title(f'Distribución de P/L - {strategy_name} ({symbol})')
            plt.xlabel('Ganancia/Pérdida')
            plt.ylabel('Frecuencia')
            plt.grid(True)
            
            profit_dist_path = f"{self.plots_dir}/profit_dist_{strategy_name}_{symbol}_{timestamp}.png"
            plt.savefig(profit_dist_path)
            plt.close()
            
            result.charts["profit_distribution"] = profit_dist_path
            
            # 5. Estadísticas mensuales
            if len(result.trades) > 0:
                # Convertir a DataFrame
                trades_df = pd.DataFrame(result.trades)
                
                # Convertir fechas
                trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
                trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
                
                # Extraer mes
                trades_df["month"] = trades_df["exit_time"].dt.strftime('%Y-%m')
                
                # Agrupar por mes
                monthly_stats = trades_df.groupby("month").agg({
                    "profit": ["sum", "count"],
                    "id": "count"
                })
                
                # Renombrar columnas
                monthly_stats.columns = ["profit", "trades", "trades_count"]
                
                # Ordenar por mes
                monthly_stats = monthly_stats.sort_index()
                
                # Plotear
                plt.figure(figsize=(12, 6))
                
                # Gráfico de barras para ganancias/pérdidas
                plt.bar(monthly_stats.index, monthly_stats["profit"], alpha=0.7, 
                      color=['green' if p > 0 else 'red' for p in monthly_stats["profit"]])
                
                plt.title(f'Rendimiento Mensual - {strategy_name} ({symbol})')
                plt.xlabel('Mes')
                plt.ylabel('Ganancia/Pérdida')
                plt.grid(True, axis='y')
                plt.xticks(rotation=45)
                
                monthly_path = f"{self.plots_dir}/monthly_{strategy_name}_{symbol}_{timestamp}.png"
                plt.savefig(monthly_path)
                plt.close()
                
                result.charts["monthly_performance"] = monthly_path
        
        except Exception as e:
            self.logger.error(f"Error al generar gráficos: {e}")
    
    async def optimize_strategy(
        self,
        strategy_class: Type[Strategy],
        param_grid: Dict[str, List[Any]],
        symbol: str,
        timeframe: str = "1h",
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_per_trade: float = 0.02,
        n_jobs: int = -1,
        metric: str = "profit_loss"
    ) -> Dict[str, Any]:
        """
        Optimizar parámetros de una estrategia mediante grid search.
        
        Args:
            strategy_class: Clase de la estrategia
            param_grid: Grid de parámetros a probar (dict de listas)
            symbol: Símbolo de trading
            timeframe: Timeframe para los datos
            start_date: Fecha de inicio
            end_date: Fecha de fin
            initial_balance: Balance inicial
            commission: Comisión por operación
            slippage: Slippage por operación
            risk_per_trade: Riesgo por operación
            n_jobs: Número de jobs para paralelización (-1 para todos)
            metric: Métrica a optimizar
            
        Returns:
            Diccionario con los mejores parámetros y resultados
        """
        self.logger.info(f"Iniciando optimización para {strategy_class.__name__} en {symbol} ({timeframe})")
        
        # Verificar que param_grid sea un diccionario de listas
        if not all(isinstance(values, list) for values in param_grid.values()):
            self.logger.error("param_grid debe ser un diccionario de listas")
            return {}
        
        # Cargar datos
        data = await self._load_historical_data(symbol, timeframe, start_date, end_date)
        if data is None or len(data) < 10:
            self.logger.error(f"Datos insuficientes para {symbol} ({timeframe})")
            return {}
        
        # Generar combinaciones de parámetros
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Número total de combinaciones
        total_combinations = len(param_combinations)
        self.logger.info(f"Evaluando {total_combinations} combinaciones de parámetros")
        
        # Ajustar n_jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        n_jobs = min(n_jobs, mp.cpu_count(), total_combinations)
        
        # Preparar resultados
        all_results = []
        
        # Si es una sola combinación o pocas, hacerlo secuencialmente
        if total_combinations <= 5:
            for i, params in enumerate(param_combinations):
                self.logger.debug(f"Evaluando combinación {i+1}/{total_combinations}: {params}")
                
                # Ejecutar backtest
                result = await self.run_backtest(
                    strategy_class,
                    params,
                    symbol,
                    timeframe,
                    start_date,
                    end_date,
                    initial_balance,
                    commission,
                    slippage,
                    risk_per_trade,
                    generate_plots=False
                )
                
                # Guardar resultados
                if result:
                    all_results.append({
                        "params": params,
                        "metrics": result.metrics
                    })
        
        # Para muchas combinaciones, usar paralelización
        else:
            # Crear función para evaluación en paralelo
            def evaluate_params(params):
                # Crear loop de asyncio para este proceso
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Ejecutar backtest
                    result = loop.run_until_complete(self.run_backtest(
                        strategy_class,
                        params,
                        symbol,
                        timeframe,
                        start_date,
                        end_date,
                        initial_balance,
                        commission,
                        slippage,
                        risk_per_trade,
                        generate_plots=False
                    ))
                    
                    # Devolver resultados
                    if result:
                        return {
                            "params": params,
                            "metrics": result.metrics
                        }
                    return None
                
                finally:
                    loop.close()
            
            # Ejecutar en paralelo
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                future_results = [executor.submit(evaluate_params, params) for params in param_combinations]
                
                # Recopilar resultados
                for i, future in enumerate(future_results):
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                        self.logger.debug(f"Completado {i+1}/{total_combinations}")
                    except Exception as e:
                        self.logger.error(f"Error en evaluación paralela: {e}")
        
        # Encontrar los mejores parámetros
        if not all_results:
            self.logger.error("No se obtuvieron resultados válidos")
            return {}
        
        # Ordenar por la métrica seleccionada
        sorted_results = sorted(all_results, key=lambda x: x["metrics"].get(metric, 0), reverse=True)
        
        # Mejores parámetros
        best_result = sorted_results[0]
        best_params = best_result["params"]
        best_metrics = best_result["metrics"]
        
        self.logger.info(f"Optimización completada. Mejores parámetros: {best_params}")
        self.logger.info(f"Mejor {metric}: {best_metrics.get(metric, 0)}")
        
        # Guardar resultados
        result_path = f"{self.results_dir}/optimization_{strategy_class.__name__}_{symbol}_{timeframe}_{int(time.time())}.json"
        
        with open(result_path, "w") as f:
            json.dump({
                "strategy": strategy_class.__name__,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat() if isinstance(start_date, datetime) else start_date,
                "end_date": end_date.isoformat() if isinstance(end_date, datetime) else end_date,
                "param_grid": param_grid,
                "best_params": best_params,
                "best_metrics": best_metrics,
                "all_results": sorted_results[:10]  # Guardar solo los 10 mejores
            }, f, indent=2)
        
        # Volver a ejecutar con los mejores parámetros para generar gráficos
        final_result = await self.run_backtest(
            strategy_class,
            best_params,
            symbol,
            timeframe,
            start_date,
            end_date,
            initial_balance,
            commission,
            slippage,
            risk_per_trade,
            generate_plots=True
        )
        
        # Devolver resultados
        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "result_path": result_path,
            "charts": final_result.charts if final_result else {}
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generar todas las combinaciones de parámetros.
        
        Args:
            param_grid: Grid de parámetros
            
        Returns:
            Lista de diccionarios con combinaciones
        """
        # Obtener nombres de parámetros y valores
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Función recursiva para generar combinaciones
        def generate_combinations(index, current_combo):
            if index == len(param_names):
                return [current_combo.copy()]
            
            result = []
            for value in param_values[index]:
                current_combo[param_names[index]] = value
                result.extend(generate_combinations(index + 1, current_combo))
            
            return result
        
        return generate_combinations(0, {})
    
    async def walk_forward_analysis(
        self,
        strategy_class: Type[Strategy],
        param_grid: Dict[str, List[Any]],
        symbol: str,
        timeframe: str = "1h",
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        window_size: int = 30,  # días
        test_size: int = 10,    # días
        step_size: int = 10,    # días
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_per_trade: float = 0.02,
        metric: str = "profit_loss"
    ) -> Dict[str, Any]:
        """
        Realizar análisis Walk-Forward para validación de estrategias.
        
        Args:
            strategy_class: Clase de la estrategia
            param_grid: Grid de parámetros a probar
            symbol: Símbolo de trading
            timeframe: Timeframe para los datos
            start_date: Fecha de inicio
            end_date: Fecha de fin
            window_size: Tamaño de la ventana de entrenamiento (días)
            test_size: Tamaño de la ventana de prueba (días)
            step_size: Tamaño del paso entre ventanas (días)
            initial_balance: Balance inicial
            commission: Comisión por operación
            slippage: Slippage por operación
            risk_per_trade: Riesgo por operación
            metric: Métrica a optimizar
            
        Returns:
            Resultados del análisis
        """
        self.logger.info(f"Iniciando análisis Walk-Forward para {strategy_class.__name__} en {symbol} ({timeframe})")
        
        # Convertir fechas si son strings
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Si no se especifican fechas, usar últimos 6 meses
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=180)
        
        # Cargar datos históricos
        data = await self._load_historical_data(symbol, timeframe, start_date, end_date)
        if data is None or len(data) < 10:
            self.logger.error(f"Datos insuficientes para {symbol} ({timeframe})")
            return {}
        
        # Generar ventanas de análisis
        windows = []
        
        # Convertir períodos a unidades de timeframe
        window_size_td = timedelta(days=window_size)
        test_size_td = timedelta(days=test_size)
        step_size_td = timedelta(days=step_size)
        
        # Generar ventanas
        current_train_start = start_date
        
        while current_train_start + window_size_td + test_size_td <= end_date:
            train_end = current_train_start + window_size_td
            test_end = train_end + test_size_td
            
            windows.append({
                "train_start": current_train_start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end
            })
            
            current_train_start += step_size_td
        
        # Si no hay suficientes ventanas, ajustar
        if len(windows) < 2:
            self.logger.warning("Período insuficiente para walk-forward. Ajustando ventanas.")
            # Dividir el período en 2 (70% train, 30% test)
            total_days = (end_date - start_date).days
            train_days = int(total_days * 0.7)
            
            windows = [{
                "train_start": start_date,
                "train_end": start_date + timedelta(days=train_days),
                "test_start": start_date + timedelta(days=train_days),
                "test_end": end_date
            }]
        
        self.logger.info(f"Analizando {len(windows)} ventanas de tiempo")
        
        # Resultados por ventana
        window_results = []
        
        # Analizar cada ventana
        for i, window in enumerate(windows):
            self.logger.info(f"Ventana {i+1}/{len(windows)}: {window['train_start']} - {window['test_end']}")
            
            # Optimizar en datos de entrenamiento
            self.logger.info(f"Optimizando en período de entrenamiento: {window['train_start']} - {window['train_end']}")
            
            opt_result = await self.optimize_strategy(
                strategy_class,
                param_grid,
                symbol,
                timeframe,
                window["train_start"],
                window["train_end"],
                initial_balance,
                commission,
                slippage,
                risk_per_trade,
                metric=metric
            )
            
            if not opt_result:
                self.logger.error(f"Optimización fallida para ventana {i+1}")
                continue
            
            best_params = opt_result["best_params"]
            
            # Probar en datos de prueba
            self.logger.info(f"Probando en período de prueba: {window['test_start']} - {window['test_end']}")
            
            test_result = await self.run_backtest(
                strategy_class,
                best_params,
                symbol,
                timeframe,
                window["test_start"],
                window["test_end"],
                initial_balance,
                commission,
                slippage,
                risk_per_trade,
                generate_plots=False
            )
            
            if not test_result:
                self.logger.error(f"Prueba fallida para ventana {i+1}")
                continue
            
            # Guardar resultados
            window_results.append({
                "window": i+1,
                "train_period": f"{window['train_start']} - {window['train_end']}",
                "test_period": f"{window['test_start']} - {window['test_end']}",
                "best_params": best_params,
                "train_metrics": opt_result["best_metrics"],
                "test_metrics": test_result.metrics
            })
        
        # Si no hay resultados, salir
        if not window_results:
            self.logger.error("No se obtuvieron resultados válidos para ninguna ventana")
            return {}
        
        # Calcular métricas agregadas
        aggregated_metrics = self._aggregate_wfa_metrics(window_results)
        
        # Guardar resultados
        result_path = f"{self.results_dir}/wfa_{strategy_class.__name__}_{symbol}_{timeframe}_{int(time.time())}.json"
        
        with open(result_path, "w") as f:
            json.dump({
                "strategy": strategy_class.__name__,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat() if isinstance(start_date, datetime) else start_date,
                "end_date": end_date.isoformat() if isinstance(end_date, datetime) else end_date,
                "param_grid": param_grid,
                "window_size": window_size,
                "test_size": test_size,
                "step_size": step_size,
                "window_results": window_results,
                "aggregated_metrics": aggregated_metrics
            }, f, indent=2)
        
        # Generar gráficos
        charts = await self._generate_wfa_plots(window_results, symbol, strategy_class.__name__)
        
        self.logger.info(f"Análisis Walk-Forward completado. Robustez: {aggregated_metrics['robustness']:.2f}")
        
        return {
            "window_results": window_results,
            "aggregated_metrics": aggregated_metrics,
            "result_path": result_path,
            "charts": charts
        }
    
    def _aggregate_wfa_metrics(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Agregar métricas de walk-forward analysis.
        
        Args:
            window_results: Resultados por ventana
            
        Returns:
            Métricas agregadas
        """
        # Extraer métricas de test
        test_metrics = [w["test_metrics"] for w in window_results]
        
        # Profit/Loss total y promedio
        total_pl = sum(m.get("profit_loss", 0) for m in test_metrics)
        avg_pl = total_pl / len(test_metrics) if test_metrics else 0
        
        # Win rates
        win_rates = [m.get("win_rate", 0) for m in test_metrics]
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
        min_win_rate = min(win_rates) if win_rates else 0
        max_win_rate = max(win_rates) if win_rates else 0
        
        # Sharpe ratios
        sharpe_ratios = [m.get("sharpe_ratio", 0) for m in test_metrics]
        avg_sharpe = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
        
        # Trades
        total_trades = sum(m.get("total_trades", 0) for m in test_metrics)
        avg_trades = total_trades / len(test_metrics) if test_metrics else 0
        
        # Robustez (porcentaje de ventanas con profit > 0)
        profitable_windows = sum(1 for m in test_metrics if m.get("profit_loss", 0) > 0)
        robustness = profitable_windows / len(test_metrics) if test_metrics else 0
        
        # Consistencia (desviación estándar de win rates)
        win_rate_std = np.std(win_rates) if len(win_rates) > 1 else 0
        consistency = 1 - (win_rate_std / avg_win_rate) if avg_win_rate > 0 else 0
        
        # Deterioro (diferencia entre train y test)
        deterioration = []
        for w in window_results:
            train_pl = w["train_metrics"].get("profit_loss", 0)
            test_pl = w["test_metrics"].get("profit_loss", 0)
            
            if train_pl > 0:
                # Deterioro como porcentaje del resultado de entrenamiento
                det = (train_pl - test_pl) / abs(train_pl)
                deterioration.append(det)
        
        avg_deterioration = sum(deterioration) / len(deterioration) if deterioration else 0
        
        return {
            "total_profit_loss": total_pl,
            "avg_profit_loss": avg_pl,
            "avg_win_rate": avg_win_rate,
            "min_win_rate": min_win_rate,
            "max_win_rate": max_win_rate,
            "win_rate_std": win_rate_std,
            "avg_sharpe_ratio": avg_sharpe,
            "total_trades": total_trades,
            "avg_trades": avg_trades,
            "robustness": robustness,
            "consistency": consistency,
            "avg_deterioration": avg_deterioration,
            "windows_count": len(window_results)
        }
    
    async def _generate_wfa_plots(
        self, 
        window_results: List[Dict[str, Any]], 
        symbol: str, 
        strategy_name: str
    ) -> Dict[str, str]:
        """
        Generar gráficos para el análisis Walk-Forward.
        
        Args:
            window_results: Resultados por ventana
            symbol: Símbolo de trading
            strategy_name: Nombre de la estrategia
            
        Returns:
            Diccionario con rutas a los gráficos
        """
        charts = {}
        timestamp = int(time.time())
        
        try:
            # 1. Profit/Loss por ventana
            plt.figure(figsize=(12, 6))
            
            # Datos para el gráfico
            window_nums = [w["window"] for w in window_results]
            train_pl = [w["train_metrics"].get("profit_loss", 0) for w in window_results]
            test_pl = [w["test_metrics"].get("profit_loss", 0) for w in window_results]
            
            # Graficar barras
            width = 0.35
            plt.bar([x - width/2 for x in window_nums], train_pl, width, label='Train', color='blue', alpha=0.7)
            plt.bar([x + width/2 for x in window_nums], test_pl, width, label='Test', color='orange', alpha=0.7)
            
            plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
            plt.xlabel('Ventana')
            plt.ylabel('Profit/Loss')
            plt.title(f'Rendimiento por Ventana - {strategy_name} ({symbol})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            wfa_pl_path = f"{self.plots_dir}/wfa_pl_{strategy_name}_{symbol}_{timestamp}.png"
            plt.savefig(wfa_pl_path)
            plt.close()
            
            charts["profit_loss_by_window"] = wfa_pl_path
            
            # 2. Win Rate por ventana
            plt.figure(figsize=(12, 6))
            
            # Datos para el gráfico
            train_wr = [w["train_metrics"].get("win_rate", 0) * 100 for w in window_results]
            test_wr = [w["test_metrics"].get("win_rate", 0) * 100 for w in window_results]
            
            # Graficar barras
            plt.bar([x - width/2 for x in window_nums], train_wr, width, label='Train', color='blue', alpha=0.7)
            plt.bar([x + width/2 for x in window_nums], test_wr, width, label='Test', color='orange', alpha=0.7)
            
            plt.xlabel('Ventana')
            plt.ylabel('Win Rate (%)')
            plt.title(f'Win Rate por Ventana - {strategy_name} ({symbol})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            wfa_wr_path = f"{self.plots_dir}/wfa_wr_{strategy_name}_{symbol}_{timestamp}.png"
            plt.savefig(wfa_wr_path)
            plt.close()
            
            charts["win_rate_by_window"] = wfa_wr_path
            
            # 3. Estabilidad de parámetros
            # Identificar parámetros comunes
            if window_results and "best_params" in window_results[0]:
                common_params = list(window_results[0]["best_params"].keys())
                
                for param_name in common_params:
                    param_values = [w["best_params"].get(param_name, None) for w in window_results]
                    
                    # Solo graficar si son valores numéricos
                    if all(isinstance(v, (int, float)) for v in param_values if v is not None):
                        plt.figure(figsize=(12, 4))
                        plt.plot(window_nums, param_values, marker='o', linestyle='-', color='purple')
                        plt.xlabel('Ventana')
                        plt.ylabel(f'Valor de {param_name}')
                        plt.title(f'Estabilidad de Parámetro: {param_name} - {strategy_name} ({symbol})')
                        plt.grid(True, alpha=0.3)
                        
                        param_path = f"{self.plots_dir}/wfa_param_{param_name}_{strategy_name}_{symbol}_{timestamp}.png"
                        plt.savefig(param_path)
                        plt.close()
                        
                        charts[f"param_stability_{param_name}"] = param_path
            
            # 4. Métricas comparativas
            plt.figure(figsize=(10, 6))
            
            # Datos para comparar train vs test
            metrics_to_compare = ["profit_loss", "sharpe_ratio", "win_rate", "max_drawdown_pct"]
            labels = ["Profit/Loss", "Sharpe Ratio", "Win Rate", "Max Drawdown"]
            
            # Calcular promedios
            train_avgs = []
            test_avgs = []
            
            for metric in metrics_to_compare:
                train_vals = [w["train_metrics"].get(metric, 0) for w in window_results]
                test_vals = [w["test_metrics"].get(metric, 0) for w in window_results]
                
                # Normalizar para visualización
                if metric == "win_rate" or metric == "max_drawdown_pct":
                    train_vals = [v * 100 for v in train_vals]
                    test_vals = [v * 100 for v in test_vals]
                
                train_avg = sum(train_vals) / len(train_vals) if train_vals else 0
                test_avg = sum(test_vals) / len(test_vals) if test_vals else 0
                
                train_avgs.append(train_avg)
                test_avgs.append(test_avg)
            
            # Graficar
            x = range(len(labels))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], train_avgs, width, label='Train', color='blue', alpha=0.7)
            plt.bar([i + width/2 for i in x], test_avgs, width, label='Test', color='orange', alpha=0.7)
            
            plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
            plt.xlabel('Métrica')
            plt.ylabel('Valor')
            plt.title(f'Comparación de Métricas - {strategy_name} ({symbol})')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            metrics_path = f"{self.plots_dir}/wfa_metrics_{strategy_name}_{symbol}_{timestamp}.png"
            plt.savefig(metrics_path)
            plt.close()
            
            charts["metrics_comparison"] = metrics_path
        
        except Exception as e:
            self.logger.error(f"Error al generar gráficos WFA: {e}")
        
        return charts


# Exportación para uso fácil
backtest_engine = BacktestEngine()