"""
Gestor de exchanges para el sistema Genesis.

Este módulo maneja la distribución de operaciones entre múltiples exchanges,
seleccionando el óptimo según criterios como liquidez y comisiones.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional

from genesis.core.base import Component
from genesis.exchanges.api_client import APIClient
from genesis.exchanges.exchange_selector import ExchangeSelector

class ExchangeManager(Component):
    """
    Manejo de múltiples exchanges y distribución de operaciones.
    
    Este componente coordina operaciones entre múltiples exchanges,
    seleccionando el óptimo para cada operación.
    """
    
    def __init__(self, exchange_configs: Dict[str, Dict[str, Any]], name: str = "exchange_manager"):
        """
        Inicializar el gestor de exchanges.
        
        Args:
            exchange_configs: Configuración para cada exchange
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self.exchanges = {}
        self.exchange_configs = exchange_configs
        self.selector = None
        
    async def start(self) -> None:
        """Iniciar el gestor de exchanges."""
        await super().start()
        
        # Inicializar clientes de exchanges
        for exchange_name, config in self.exchange_configs.items():
            try:
                self.exchanges[exchange_name] = APIClient(exchange_name, config)
                self.logger.info(f"Exchange inicializado: {exchange_name}")
            except Exception as e:
                self.logger.error(f"Error al inicializar {exchange_name}: {e}")
        
        # Inicializar selector
        self.selector = ExchangeSelector(self.exchanges)
        
        self.logger.info(f"Gestor de exchanges iniciado con {len(self.exchanges)} exchanges")
        
    async def stop(self) -> None:
        """Detener el gestor de exchanges."""
        self.logger.info("Deteniendo gestor de exchanges")
        await super().stop()
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        if event_type == "trade.request":
            await self._handle_trade_request(data)
        elif event_type == "system.status_request":
            await self._handle_status_request()
    
    async def get_best_exchange(self, trading_pair: str) -> Optional[str]:
        """
        Devuelve el mejor exchange para operar en un par de trading.
        
        Args:
            trading_pair: Par de trading (ej: 'BTC/USDT')
            
        Returns:
            Nombre del mejor exchange o None
        """
        if not self.selector:
            self.logger.error("Selector no inicializado")
            return None
            
        return await self.selector.get_best_exchange(trading_pair)
    
    async def execute_trade(self, trading_pair: str, side: str, amount: float, 
                          price: Optional[float] = None, specific_exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta una operación en el mejor exchange disponible.
        
        Args:
            trading_pair: Par de trading (ej: 'BTC/USDT')
            side: Dirección ('buy' o 'sell')
            amount: Cantidad a operar
            price: Precio límite (opcional)
            specific_exchange: Exchange específico a utilizar (opcional)
            
        Returns:
            Resultado de la operación
        """
        # Usar exchange específico o buscar el mejor
        exchange_name = specific_exchange
        if not exchange_name:
            exchange_name = await self.get_best_exchange(trading_pair)
        
        if not exchange_name or exchange_name not in self.exchanges:
            self.logger.error(f"No se encontró exchange adecuado para {trading_pair}")
            return {"status": "error", "message": "No suitable exchange found"}
        
        # Ejecutar la operación
        client = self.exchanges[exchange_name]
        
        # Convertir el método síncrono a asíncrono
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: client.place_order(trading_pair, side, amount, price)
        )
        
        # Emitir evento de resultado
        if result.get("status") == "ok":
            await self.emit_event("trade.executed", {
                "trading_pair": trading_pair,
                "side": side,
                "amount": amount,
                "price": price,
                "exchange": exchange_name,
                "order_id": result.get("order_id")
            })
        else:
            await self.emit_event("trade.failed", {
                "trading_pair": trading_pair,
                "side": side,
                "amount": amount,
                "exchange": exchange_name,
                "error": result.get("message")
            })
            
        return result
    
    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener balances en todos los exchanges.
        
        Returns:
            Diccionario con balances por exchange
        """
        balances = {}
        for name, client in self.exchanges.items():
            try:
                # Convertir método síncrono a asíncrono
                loop = asyncio.get_event_loop()
                balance = await loop.run_in_executor(None, client.get_balance)
                balances[name] = balance
            except Exception as e:
                self.logger.error(f"Error obteniendo balance de {name}: {e}")
                balances[name] = {}
                
        return balances
    
    async def _handle_trade_request(self, data: Dict[str, Any]) -> None:
        """
        Manejar solicitud de trading.
        
        Args:
            data: Datos de la solicitud
        """
        trading_pair = data.get("trading_pair")
        side = data.get("side")
        amount = data.get("amount")
        price = data.get("price")
        exchange = data.get("exchange")
        
        if not all([trading_pair, side, amount]):
            await self.emit_event("trade.error", {
                "message": "Missing required parameters",
                "data": data
            })
            return
            
        result = await self.execute_trade(trading_pair, side, amount, price, exchange)
        
        # El resultado ya es emitido en execute_trade
    
    async def _handle_status_request(self) -> None:
        """Manejar solicitud de estado del sistema."""
        statuses = []
        for name, client in self.exchanges.items():
            # Convertir método síncrono a asíncrono
            loop = asyncio.get_event_loop()
            latency = await loop.run_in_executor(None, client.get_latency)
            statuses.append({
                "exchange": name,
                "status": "online" if latency < 5000 else "slow",
                "latency": latency
            })
        
        await self.emit_event("system.exchange_status", {
            "exchanges": statuses
        })
        
    def get_exchange_list(self) -> List[str]:
        """
        Obtener lista de exchanges disponibles.
        
        Returns:
            Lista de nombres de exchanges
        """
        return list(self.exchanges.keys())