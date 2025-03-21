"""
Gestor de datos de mercado para el sistema Genesis.

Este módulo proporciona funcionalidades para obtener, procesar y analizar
datos de mercado de diferentes exchanges.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

from genesis.core.base import Component

class MarketData(Component):
    """
    Gestor de datos de mercado con manejo de concurrencia y análisis técnico.
    
    Este componente se encarga de obtener datos de diferentes exchanges,
    procesarlos y calcular indicadores técnicos.
    """
    
    def __init__(
        self, 
        update_interval: float = 5.0, 
        max_concurrent: int = 50,
        name: str = "market_data"
    ):
        """
        Inicializar el gestor de datos de mercado.
        
        Args:
            update_interval: Intervalo en segundos entre actualizaciones
            max_concurrent: Número máximo de operaciones concurrentes
            name: Nombre del componente
        """
        super().__init__(name)
        self.api_clients = {}
        self.update_interval = max(1.0, update_interval)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.data = {}
        self.active_symbols = set()
        self.update_task = None
        self.logger = logging.getLogger(__name__)
        
    def register_client(self, exchange_name: str, api_client: Any) -> None:
        """
        Registrar un cliente API para un exchange.
        
        Args:
            exchange_name: Nombre del exchange
            api_client: Cliente API para el exchange
        """
        self.api_clients[exchange_name] = api_client
        self.logger.info(f"Cliente API registrado para {exchange_name}")
        
    async def start(self) -> None:
        """Iniciar el gestor de datos de mercado."""
        await super().start()
        self.logger.info("Gestor de datos de mercado iniciado")
        
    async def stop(self) -> None:
        """Detener el gestor de datos de mercado."""
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
                
        await super().stop()
        self.logger.info("Gestor de datos de mercado detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        if event_type == "market.start_tracking":
            symbols = data.get("symbols", [])
            if symbols:
                self.start_update_loop(symbols)
                
        elif event_type == "market.stop_tracking":
            symbols = data.get("symbols", [])
            if symbols:
                self.stop_tracking(symbols)
                
        elif event_type == "market.request_data":
            symbol = data.get("symbol")
            exchange = data.get("exchange")
            
            if symbol and exchange:
                market_data = self.get_market_data(symbol, exchange)
                await self.emit_event("market.data_response", {
                    "symbol": symbol,
                    "exchange": exchange,
                    "data": market_data,
                    "request_id": data.get("request_id")
                })
                
    async def fetch_symbol_data(self, symbol: str, exchange_name: str, client: Any) -> None:
        """
        Obtiene datos del mercado de un símbolo desde un exchange.
        
        Args:
            symbol: Símbolo del par de trading
            exchange_name: Nombre del exchange
            client: Cliente API del exchange
        """
        async with self.semaphore:
            try:
                # Convertir método síncrono a asíncrono si es necesario
                if hasattr(client, "fetch_market_data") and asyncio.iscoroutinefunction(client.fetch_market_data):
                    data = await client.fetch_market_data(symbol)
                elif hasattr(client, "fetch_market_data"):
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(None, lambda: client.fetch_market_data(symbol))
                else:
                    self.logger.error(f"Cliente para {exchange_name} no tiene método fetch_market_data")
                    return
                    
                # Validar datos
                if not data or "price" not in data or data["price"] <= 0:
                    self.logger.error(f"Datos inválidos de {exchange_name} para {symbol}")
                    return
                    
                # Almacenar datos
                self.data.setdefault(symbol, {})[exchange_name] = data
                self.logger.debug(f"Actualizado {symbol} en {exchange_name}: {data['price']}")
                
                # Emitir evento
                await self.emit_event("market.tick", {
                    "symbol": symbol,
                    "exchange": exchange_name,
                    "price": data["price"],
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                self.logger.warning(f"Error en {exchange_name} para {symbol}: {e}")
                
    def start_update_loop(self, symbols: List[str]) -> None:
        """
        Inicia la actualización constante de datos de los símbolos dados.
        
        Args:
            symbols: Lista de símbolos a monitorear
        """
        self.active_symbols.update(symbols)
        if self.update_task is None or self.update_task.done():
            self.update_task = asyncio.create_task(self.update_loop())
            self.logger.info(f"Iniciado loop de actualización para {len(self.active_symbols)} símbolos")
        else:
            self.logger.info(f"Símbolos añadidos al loop de actualización: {len(symbols)}")
            
    def stop_tracking(self, symbols: Optional[List[str]] = None) -> None:
        """
        Detiene el seguimiento de símbolos específicos o todos.
        
        Args:
            symbols: Lista de símbolos a dejar de seguir, o None para todos
        """
        if symbols is None:
            self.active_symbols.clear()
            if self.update_task:
                self.update_task.cancel()
                self.update_task = None
            self.logger.info("Detenido seguimiento de todos los símbolos")
        else:
            for symbol in symbols:
                self.active_symbols.discard(symbol)
            self.logger.info(f"Detenido seguimiento de {len(symbols)} símbolos")
            
    async def update_loop(self) -> None:
        """Loop principal para actualizar datos de todos los símbolos activos."""
        while self.running and self.active_symbols:
            tasks = []
            for symbol in self.active_symbols:
                for exchange_name, client in self.api_clients.items():
                    tasks.append(self.fetch_symbol_data(symbol, exchange_name, client))
                    
            if tasks:
                # Ejecutar todas las tareas en paralelo
                await asyncio.gather(*tasks, return_exceptions=True)
                
            # Pausa entre actualizaciones
            await asyncio.sleep(self.update_interval)
            
    def get_price(self, symbol: str, exchange_name: str) -> float:
        """
        Obtener el precio más reciente de un símbolo.
        
        Args:
            symbol: Símbolo del par de trading
            exchange_name: Nombre del exchange
            
        Returns:
            Precio actual o 0 si no hay datos
        """
        return self.data.get(symbol, {}).get(exchange_name, {}).get("price", 0)
        
    def get_ohlcv(self, symbol: str, exchange_name: str) -> List:
        """
        Obtener datos OHLCV (velas) de un símbolo.
        
        Args:
            symbol: Símbolo del par de trading
            exchange_name: Nombre del exchange
            
        Returns:
            Lista de velas OHLCV o lista vacía si no hay datos
        """
        return self.data.get(symbol, {}).get(exchange_name, {}).get("ohlcv", [])
        
    def get_market_data(self, symbol: str, exchange_name: str) -> Dict[str, Any]:
        """
        Obtener todos los datos disponibles para un símbolo.
        
        Args:
            symbol: Símbolo del par de trading
            exchange_name: Nombre del exchange
            
        Returns:
            Diccionario con datos del mercado o diccionario vacío
        """
        return self.data.get(symbol, {}).get(exchange_name, {})
        
    def get_atr(self, symbol: str, exchange_name: str, period: int = 14) -> float:
        """
        Calcula el ATR (Average True Range) desde los datos OHLCV.
        
        Args:
            symbol: Símbolo del par de trading
            exchange_name: Nombre del exchange
            period: Período para el cálculo
            
        Returns:
            Valor del ATR o 0 si no hay suficientes datos
        """
        ohlcv = self.get_ohlcv(symbol, exchange_name)
        if len(ohlcv) < period + 1:
            return 0.0
            
        try:
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # Calcular TR (True Range)
            high = df["high"]
            low = df["low"]
            close = df["close"]
            prev_close = close.shift(1)
            
            # El TR es el máximo de: (high-low), |high-prev_close|, |low-prev_close|
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            
            # ATR es la media móvil del TR
            atr = tr.rolling(window=period).mean().iloc[-1]
            return round(atr, 6) if not np.isnan(atr) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculando ATR para {symbol} en {exchange_name}: {e}")
            return 0.0
            
    def get_volatility(self, symbol: str, exchange_name: str, window: int = 20) -> float:
        """
        Calcula la volatilidad (desviación estándar de retornos).
        
        Args:
            symbol: Símbolo del par de trading
            exchange_name: Nombre del exchange
            window: Ventana para el cálculo
            
        Returns:
            Volatilidad o 0 si no hay suficientes datos
        """
        ohlcv = self.get_ohlcv(symbol, exchange_name)
        if len(ohlcv) < window + 1:
            return 0.0
            
        try:
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # Calcular retornos
            returns = df["close"].pct_change().dropna()
            
            # Calcular desviación estándar
            volatility = returns.tail(window).std()
            return round(float(volatility), 6) if not np.isnan(volatility) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculando volatilidad para {symbol} en {exchange_name}: {e}")
            return 0.0
            
    async def get_all_prices(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener precios actuales de todos los símbolos en seguimiento.
        
        Returns:
            Diccionario con precios por símbolo y exchange
        """
        result = {}
        for symbol in self.active_symbols:
            result[symbol] = {}
            for exchange_name in self.api_clients:
                price = self.get_price(symbol, exchange_name)
                if price > 0:
                    result[symbol][exchange_name] = price
                    
        return result

class SlippageController:
    """
    Controlador de slippage para validar condiciones de ejecución.
    
    Esta clase valida el slippage (desviación entre precio esperado y ejecutado)
    basado en condiciones del mercado como ATR.
    """
    
    def __init__(self, base_max_slippage: float, market_data: MarketData):
        """
        Inicializar el controlador de slippage.
        
        Args:
            base_max_slippage: Slippage máximo base
            market_data: Instancia de MarketData
        """
        self.base_max_slippage = max(0.0001, min(0.05, base_max_slippage))  # Entre 0.01% y 5%
        self.market_data = market_data
        self.logger = logging.getLogger(__name__)

    def get_dynamic_slippage(self, symbol: str, exchange_name: str) -> float:
        """
        Calcula el slippage máximo dinámicamente basado en ATR.
        
        Args:
            symbol: Símbolo del par de trading
            exchange_name: Nombre del exchange
            
        Returns:
            Slippage máximo permitido
        """
        atr = self.market_data.get_atr(symbol, exchange_name)
        price = self.market_data.get_price(symbol, exchange_name)
        
        # Usar ATR relativo al precio actual
        atr_percent = atr / price if price > 0 else 0
        
        # Aumentar slippage permitido en función de ATR
        return min(self.base_max_slippage * (1 + 10 * atr_percent), 0.01)  # Máximo 1%

    def validate_slippage(
        self, 
        entry_price: float, 
        execution_price: float, 
        symbol: str, 
        exchange_name: str
    ) -> bool:
        """
        Valida si el slippage está dentro de límites aceptables.
        
        Args:
            entry_price: Precio de entrada esperado
            execution_price: Precio de ejecución real
            symbol: Símbolo del par de trading
            exchange_name: Nombre del exchange
            
        Returns:
            True si el slippage es aceptable, False en caso contrario
        """
        if entry_price <= 0 or execution_price <= 0:
            self.logger.warning(f"Precios inválidos para validar slippage: {entry_price}, {execution_price}")
            return False
            
        max_slippage = self.get_dynamic_slippage(symbol, exchange_name)
        real_slippage = abs(execution_price - entry_price) / entry_price
        
        if real_slippage > max_slippage:
            self.logger.warning(
                f"Slippage alto en {symbol}: {real_slippage:.6f} > {max_slippage:.6f}"
            )
            return False
            
        return True