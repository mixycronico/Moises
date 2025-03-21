"""
API de backtesting para el sistema Genesis.

Este módulo proporciona endpoints y funciones para acceder
a las capacidades de backtesting desde interfaces externas.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from genesis.core.base import Component
from genesis.backtesting.engine import BacktestEngine

class BacktestAPI(Component):
    """
    API para el motor de backtesting.
    
    Este componente proporciona una interfaz de alto nivel para ejecutar
    backtests y gestionar sus resultados, accesible desde componentes
    externos como la API REST y la UI.
    """
    
    def __init__(
        self,
        backtest_engine: BacktestEngine,
        name: str = "backtest_api"
    ):
        """
        Inicializar la API de backtesting.
        
        Args:
            backtest_engine: Motor de backtesting
            name: Nombre del componente
        """
        super().__init__(name)
        self.backtest_engine = backtest_engine
        self.logger = logging.getLogger(__name__)
        self.cached_results = {}
        
    async def start(self) -> None:
        """Iniciar la API de backtesting."""
        await super().start()
        self.logger.info("API de backtesting iniciada")
        
    async def stop(self) -> None:
        """Detener la API de backtesting."""
        await super().stop()
        self.logger.info("API de backtesting detenida")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        if event_type == "api.backtest.request":
            # Recibimos una solicitud de backtest desde la API REST
            request_id = data.get("request_id")
            
            try:
                # Validar y procesar parámetros
                strategy_name = data.get("strategy_name")
                symbol = data.get("symbol")
                timeframe = data.get("timeframe", "1d")
                start_date = data.get("start_date")
                end_date = data.get("end_date", datetime.now().strftime("%Y-%m-%d"))
                params = data.get("params", {})
                
                if not all([strategy_name, symbol, start_date]):
                    raise ValueError("Faltan parámetros obligatorios para el backtest")
                
                # Ejecutar backtest
                result = await self.run_backtest(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    params=params
                )
                
                # Responder con el resultado
                await self.emit_event("api.backtest.response", {
                    "success": True,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "result": result,
                    "request_id": request_id
                })
                
            except Exception as e:
                self.logger.error(f"Error en API de backtest: {e}")
                await self.emit_event("api.backtest.response", {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                })
                
        elif event_type == "api.backtest.optimize":
            # Recibimos una solicitud de optimización desde la API REST
            request_id = data.get("request_id")
            
            try:
                # Validar y procesar parámetros
                strategy_name = data.get("strategy_name")
                symbol = data.get("symbol")
                timeframe = data.get("timeframe", "1d")
                start_date = data.get("start_date")
                end_date = data.get("end_date", datetime.now().strftime("%Y-%m-%d"))
                param_grid = data.get("param_grid", {})
                metric = data.get("metric", "sharpe_ratio")
                
                if not all([strategy_name, symbol, start_date, param_grid]):
                    raise ValueError("Faltan parámetros obligatorios para la optimización")
                
                # Ejecutar optimización
                result, best_params = await self.optimize_strategy(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    param_grid=param_grid,
                    metric=metric
                )
                
                # Responder con el resultado
                await self.emit_event("api.backtest.optimize.response", {
                    "success": True,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "best_params": best_params,
                    "result": result,
                    "request_id": request_id
                })
                
            except Exception as e:
                self.logger.error(f"Error en optimización de backtest: {e}")
                await self.emit_event("api.backtest.optimize.response", {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                })
                
        elif event_type == "api.backtest.list":
            # Listar resultados de backtests disponibles
            request_id = data.get("request_id")
            
            try:
                # Obtener lista de resultados
                results = self.list_backtests()
                
                # Responder con la lista
                await self.emit_event("api.backtest.list.response", {
                    "success": True,
                    "results": results,
                    "request_id": request_id
                })
                
            except Exception as e:
                self.logger.error(f"Error al listar backtests: {e}")
                await self.emit_event("api.backtest.list.response", {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                })
                
        elif event_type == "api.backtest.get":
            # Obtener un resultado específico
            request_id = data.get("request_id")
            
            try:
                # Obtener parámetros
                strategy_name = data.get("strategy_name")
                symbol = data.get("symbol")
                
                if not all([strategy_name, symbol]):
                    raise ValueError("Faltan parámetros obligatorios para obtener el backtest")
                
                # Obtener resultado
                result = self.get_backtest_result(strategy_name, symbol)
                
                # Responder con el resultado
                await self.emit_event("api.backtest.get.response", {
                    "success": True,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "result": result,
                    "request_id": request_id
                })
                
            except Exception as e:
                self.logger.error(f"Error al obtener backtest: {e}")
                await self.emit_event("api.backtest.get.response", {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                })
                
        elif event_type == "api.backtest.plot":
            # Generar gráfico de resultados
            request_id = data.get("request_id")
            
            try:
                # Obtener parámetros
                strategy_name = data.get("strategy_name")
                symbol = data.get("symbol")
                save = data.get("save", True)
                
                if not all([strategy_name, symbol]):
                    raise ValueError("Faltan parámetros obligatorios para generar el gráfico")
                
                # Generar gráfico
                result = await self.generate_plot(strategy_name, symbol, save)
                
                # Responder con la ruta del gráfico
                await self.emit_event("api.backtest.plot.response", {
                    "success": True,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "plot_path": result,
                    "request_id": request_id
                })
                
            except Exception as e:
                self.logger.error(f"Error al generar gráfico: {e}")
                await self.emit_event("api.backtest.plot.response", {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                })
                
    async def run_backtest(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar un backtest.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo de trading
            timeframe: Intervalo de tiempo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            params: Parámetros para la estrategia
            
        Returns:
            Resultados del backtest
        """
        self.logger.info(f"Ejecutando backtest para {strategy_name} en {symbol}")
        
        try:
            # Ejecutar backtest a través del motor
            result = await self.backtest_engine.run_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                params=params
            )
            
            # Cachear resultado
            key = f"{strategy_name}_{symbol}"
            self.cached_results[key] = {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "params": params,
                "timestamp": datetime.now().isoformat(),
                "result": result
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en backtest: {e}")
            raise
            
    async def optimize_strategy(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        param_grid: Dict[str, List],
        metric: str = "sharpe_ratio"
    ) -> tuple:
        """
        Optimizar parámetros de una estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo de trading
            timeframe: Intervalo de tiempo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            param_grid: Grid de parámetros a probar
            metric: Métrica a optimizar
            
        Returns:
            Tupla de (mejor resultado, mejores parámetros)
        """
        self.logger.info(f"Optimizando {strategy_name} en {symbol}")
        
        try:
            # Obtener datos históricos
            df = await self.backtest_engine.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                raise ValueError(f"No se encontraron datos para {symbol}")
                
            # Obtener estrategia
            strategy = await self.backtest_engine.get_strategy(strategy_name)
            
            if strategy is None:
                raise ValueError(f"Estrategia '{strategy_name}' no encontrada")
                
            # Optimizar
            best_result, best_params = await self.backtest_engine.optimize_strategy(
                strategy=strategy,
                df=df,
                param_grid=param_grid,
                metric=metric
            )
            
            # Cachear resultado
            key = f"{strategy_name}_{symbol}_optimized"
            self.cached_results[key] = {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "best_params": best_params,
                "timestamp": datetime.now().isoformat(),
                "result": best_result
            }
            
            return best_result, best_params
            
        except Exception as e:
            self.logger.error(f"Error en optimización: {e}")
            raise
            
    def list_backtests(self) -> List[Dict[str, Any]]:
        """
        Listar los backtests disponibles.
        
        Returns:
            Lista de metadatos de backtests
        """
        results = []
        for key, data in self.cached_results.items():
            # Incluir solo metadatos, no los resultados completos
            results.append({
                "id": key,
                "strategy_name": data.get("strategy_name"),
                "symbol": data.get("symbol"),
                "timeframe": data.get("timeframe"),
                "start_date": data.get("start_date"),
                "end_date": data.get("end_date"),
                "timestamp": data.get("timestamp"),
                "params": data.get("params"),
                "metrics": {
                    "total_return": data.get("result", {}).get("total_return", 0),
                    "sharpe_ratio": data.get("result", {}).get("sharpe_ratio", 0),
                    "max_drawdown": data.get("result", {}).get("max_drawdown", 0),
                    "num_trades": data.get("result", {}).get("num_trades", 0),
                    "win_rate": data.get("result", {}).get("win_rate", 0)
                }
            })
            
        return results
        
    def get_backtest_result(self, strategy_name: str, symbol: str) -> Dict[str, Any]:
        """
        Obtener un resultado de backtest específico.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo de trading
            
        Returns:
            Resultado del backtest
        """
        key = f"{strategy_name}_{symbol}"
        if key not in self.cached_results:
            # Intentar con el resultado optimizado
            key = f"{strategy_name}_{symbol}_optimized"
            if key not in self.cached_results:
                raise ValueError(f"No se encontró resultado para {strategy_name} en {symbol}")
                
        return self.cached_results[key]
        
    async def generate_plot(
        self,
        strategy_name: str,
        symbol: str,
        save: bool = True
    ) -> Optional[str]:
        """
        Generar un gráfico de resultados de backtest.
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo de trading
            save: Si se debe guardar el gráfico
            
        Returns:
            Ruta al gráfico guardado o None
        """
        try:
            # Verificar si tenemos el backtest
            key = f"{strategy_name}_{symbol}"
            if key not in self.cached_results and f"{key}_optimized" not in self.cached_results:
                # Intentar obtener resultado del motor
                if f"{key}" not in self.backtest_engine.results:
                    raise ValueError(f"No se encontró resultado para {strategy_name} en {symbol}")
                    
            # Generar gráfico
            plot_path = self.backtest_engine.plot_results(
                strategy_name=strategy_name, 
                symbol=symbol,
                show=False,
                save=save
            )
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error al generar gráfico: {e}")
            raise