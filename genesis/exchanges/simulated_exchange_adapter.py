"""
Adaptador para integrar el simulador de exchange con la arquitectura del sistema.

Este módulo proporciona un adaptador que implementa las interfaces de 
las clases de exchange existentes, permitiendo usar el simulador 
sin modificar el resto del código.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Coroutine

from genesis.simulators import (
    ExchangeSimulator, 
    ExchangeSimulatorFactory,
    MarketPattern, 
    MarketEventType
)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Genesis.SimulatedExchangeAdapter")

class SimulatedExchangeAdapter:
    """
    Adaptador para el simulador de exchange.
    
    Este adaptador proporciona la misma interfaz que las clases de exchange
    existentes, permitiendo usar el simulador sin modificar el resto del código.
    """
    
    def __init__(self, exchange_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar adaptador para el simulador.
        
        Args:
            exchange_id: Identificador del exchange
            config: Configuración opcional
        """
        self.exchange_id = exchange_id
        self.config = config or {}
        self.logger = logger.getChild(f"Adapter.{exchange_id}")
        self.simulator = None
        self.is_connected = False
        self.subscriptions = set()
        self.callbacks = {}  # channel -> callback
        self.symbols = set()
        self.start_time = time.time()
        self.message_count = 0
        self.error_count = 0
        self.initialized = False
        
    async def initialize(self) -> Dict[str, Any]:
        """
        Inicializar adaptador y simulador.
        
        Returns:
            Dict con resultado de inicialización
        """
        self.logger.info(f"Inicializando adaptador para simulador {self.exchange_id}...")
        
        # Crear simulador
        self.simulator = await ExchangeSimulatorFactory.create_simulator(self.exchange_id, self.config)
        self.initialized = True
        
        # Obtener símbolos disponibles
        exchange_info = await self.simulator.get_exchange_info()
        self.symbols = set(exchange_info.get("symbols", []))
        
        self.logger.info(f"Adaptador inicializado con {len(self.symbols)} símbolos")
        
        return {
            "success": True,
            "message": "Adaptador inicializado correctamente",
            "exchange_id": self.exchange_id,
            "symbols": list(self.symbols)
        }
        
    async def connect(self) -> Dict[str, Any]:
        """
        Conectar al exchange simulado.
        
        Returns:
            Dict con resultado de conexión
        """
        self.logger.info(f"Conectando a simulador {self.exchange_id}...")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        # El simulador ya está "conectado" al inicializarse
        self.is_connected = True
        
        return {
            "success": True,
            "message": "Conexión establecida con simulador",
            "exchange_id": self.exchange_id,
            "transmuted": False
        }
        
    async def close(self) -> Dict[str, Any]:
        """
        Cerrar conexión con exchange simulado.
        
        Returns:
            Dict con resultado de cierre
        """
        self.logger.info(f"Cerrando conexión con simulador {self.exchange_id}...")
        
        if self.simulator:
            await self.simulator.shutdown()
            
        self.is_connected = False
        
        return {
            "success": True,
            "message": "Conexión cerrada correctamente",
            "exchange_id": self.exchange_id
        }
        
    async def subscribe(self, channels: List[str]) -> Dict[str, Any]:
        """
        Suscribirse a canales en el exchange simulado.
        
        Args:
            channels: Lista de canales
            
        Returns:
            Dict con resultado de suscripción
        """
        self.logger.info(f"Suscribiendo a canales en {self.exchange_id}: {channels}")
        
        # Verificar conexión
        if not self.is_connected:
            await self.connect()
            
        # Registrar suscripciones
        success_channels = []
        
        for channel in channels:
            try:
                # Analizar canal para determinar símbolo
                parts = channel.split(':')
                
                if len(parts) >= 2:
                    channel_type = parts[0]
                    symbol = parts[1].upper()
                    
                    # Crear callback para este canal
                    async def message_callback(message):
                        # Procesar mensaje y pasarlo al callback registrado para este canal
                        if channel in self.callbacks:
                            callback = self.callbacks[channel]
                            await callback(message)
                            self.message_count += 1
                            
                    # Suscribirse al canal en el simulador
                    await self.simulator.subscribe(channel, message_callback)
                    self.subscriptions.add(channel)
                    success_channels.append(channel)
                    
            except Exception as e:
                self.logger.error(f"Error suscribiendo a canal {channel}: {e}")
                self.error_count += 1
                
        return {
            "success": True,
            "subscribed_channels": success_channels,
            "message": f"Suscrito a {len(success_channels)} canales"
        }
        
    async def unsubscribe(self, channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Cancelar suscripción a canales.
        
        Args:
            channels: Lista de canales (None = todos)
            
        Returns:
            Dict con resultado de cancelación
        """
        channels_to_unsubscribe = channels if channels is not None else list(self.subscriptions)
        
        self.logger.info(f"Cancelando suscripción a canales en {self.exchange_id}: {channels_to_unsubscribe}")
        
        unsubscribed = []
        
        for channel in channels_to_unsubscribe:
            if channel in self.subscriptions:
                try:
                    # Cancelar suscripción en el simulador
                    await self.simulator.unsubscribe(channel)
                    self.subscriptions.remove(channel)
                    self.callbacks.pop(channel, None)
                    unsubscribed.append(channel)
                    
                except Exception as e:
                    self.logger.error(f"Error cancelando suscripción a {channel}: {e}")
                    self.error_count += 1
                    
        return {
            "success": True,
            "unsubscribed_channels": unsubscribed,
            "message": f"Cancelada suscripción a {len(unsubscribed)} canales"
        }
        
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener ticker para un símbolo.
        
        Args:
            symbol: Símbolo (ej: "BTC/USDT")
            
        Returns:
            Dict con datos de ticker
        """
        self.logger.debug(f"Obteniendo ticker para {symbol}")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            ticker = await self.simulator.get_ticker(symbol)
            return ticker
            
        except Exception as e:
            self.logger.error(f"Error obteniendo ticker para {symbol}: {e}")
            self.error_count += 1
            
            # Reintento de transmutación
            try:
                await asyncio.sleep(0.5)
                ticker = await self.simulator.get_ticker(symbol)
                ticker["transmuted"] = True
                return ticker
                
            except Exception as e2:
                self.logger.error(f"Error en transmutación de ticker para {symbol}: {e2}")
                raise e  # Propagar error original
                
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Obtener libro de órdenes para un símbolo.
        
        Args:
            symbol: Símbolo (ej: "BTC/USDT")
            limit: Número máximo de niveles
            
        Returns:
            Dict con libro de órdenes
        """
        self.logger.debug(f"Obteniendo orderbook para {symbol}")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            orderbook = await self.simulator.get_orderbook(symbol, limit)
            return orderbook
            
        except Exception as e:
            self.logger.error(f"Error obteniendo orderbook para {symbol}: {e}")
            self.error_count += 1
            
            # Reintento de transmutación
            try:
                await asyncio.sleep(0.5)
                orderbook = await self.simulator.get_orderbook(symbol, limit)
                orderbook["transmuted"] = True
                return orderbook
                
            except Exception as e2:
                self.logger.error(f"Error en transmutación de orderbook para {symbol}: {e2}")
                raise e  # Propagar error original
                
    async def get_candles(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener velas para un símbolo y timeframe.
        
        Args:
            symbol: Símbolo (ej: "BTC/USDT")
            timeframe: Período de tiempo (ej: "1m", "5m", "1h", "1d")
            limit: Número máximo de velas
            
        Returns:
            Lista de velas OHLCV
        """
        self.logger.debug(f"Obteniendo velas para {symbol} ({timeframe})")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            candles = await self.simulator.get_candles(symbol, timeframe, limit)
            return candles
            
        except Exception as e:
            self.logger.error(f"Error obteniendo velas para {symbol}: {e}")
            self.error_count += 1
            
            # No hay transmutación para candles
            raise e
                
    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Colocar una orden en el exchange simulado.
        
        Args:
            order_data: Datos de la orden
                symbol: Símbolo (ej: "BTC/USDT")
                side: Lado ("buy" o "sell")
                type: Tipo de orden ("market", "limit", etc.)
                quantity: Cantidad a comprar/vender
                price: Precio (para órdenes limit)
                
        Returns:
            Dict con resultado de la operación
        """
        self.logger.info(f"Colocando orden en {self.exchange_id}: {order_data}")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            result = await self.simulator.place_order(order_data)
            return result
            
        except Exception as e:
            self.logger.error(f"Error colocando orden: {e}")
            self.error_count += 1
            
            # Reintento de transmutación
            try:
                await asyncio.sleep(0.5)
                result = await self.simulator.place_order(order_data)
                if "order" in result:
                    result["order"]["transmuted"] = True
                return result
                
            except Exception as e2:
                self.logger.error(f"Error en transmutación de orden: {e2}")
                raise e  # Propagar error original
                
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancelar una orden existente.
        
        Args:
            order_id: ID de la orden
            
        Returns:
            Dict con resultado de la operación
        """
        self.logger.info(f"Cancelando orden {order_id} en {self.exchange_id}")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            result = await self.simulator.cancel_order(order_id)
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelando orden {order_id}: {e}")
            self.error_count += 1
            
            # Reintento de transmutación
            try:
                await asyncio.sleep(0.5)
                result = await self.simulator.cancel_order(order_id)
                if "order" in result:
                    result["order"]["transmuted"] = True
                return result
                
            except Exception as e2:
                self.logger.error(f"Error en transmutación de cancelación: {e2}")
                raise e  # Propagar error original
                
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener órdenes filtradas.
        
        Args:
            symbol: Símbolo opcional para filtrar
            status: Estado opcional para filtrar
            
        Returns:
            Dict con lista de órdenes
        """
        self.logger.debug(f"Obteniendo órdenes para {symbol if symbol else 'todos los símbolos'}")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            result = await self.simulator.get_orders(symbol, status)
            return result
            
        except Exception as e:
            self.logger.error(f"Error obteniendo órdenes: {e}")
            self.error_count += 1
            
            # No hay transmutación para órdenes
            raise e
                
    async def set_market_pattern(self, symbol: str, pattern: MarketPattern) -> Dict[str, Any]:
        """
        Establecer patrón de mercado para un símbolo.
        
        Args:
            symbol: Símbolo
            pattern: Patrón de mercado
            
        Returns:
            Dict con resultado de la operación
        """
        self.logger.info(f"Estableciendo patrón {pattern.name} para {symbol} en {self.exchange_id}")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            result = await self.simulator.set_market_pattern(symbol, pattern)
            return result
            
        except Exception as e:
            self.logger.error(f"Error estableciendo patrón: {e}")
            self.error_count += 1
            raise e
                
    async def add_market_event(self, event_type: MarketEventType, symbol: str, 
                              impact: float = 0.02, delay: float = 0) -> Dict[str, Any]:
        """
        Añadir evento de mercado para un símbolo.
        
        Args:
            event_type: Tipo de evento
            symbol: Símbolo
            impact: Impacto del evento (0-1)
            delay: Retraso en segundos
            
        Returns:
            Dict con resultado de la operación
        """
        self.logger.info(f"Añadiendo evento {event_type.name} para {symbol} en {self.exchange_id}")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            result = await self.simulator.add_market_event(event_type, symbol, impact, delay)
            return result
            
        except Exception as e:
            self.logger.error(f"Error añadiendo evento: {e}")
            self.error_count += 1
            raise e
                
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Obtener información del exchange.
        
        Returns:
            Dict con información del exchange
        """
        self.logger.debug(f"Obteniendo información de {self.exchange_id}")
        
        # Verificar inicialización
        if not self.initialized:
            await self.initialize()
            
        try:
            info = await self.simulator.get_exchange_info()
            
            # Añadir estadísticas del adaptador
            info["adapter_stats"] = {
                "uptime": time.time() - self.start_time,
                "message_count": self.message_count,
                "error_count": self.error_count,
                "subscription_count": len(self.subscriptions)
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error obteniendo información: {e}")
            self.error_count += 1
            raise e
                
    async def register_callback(self, channel: str, callback: Callable[[Dict[str, Any]], Coroutine]) -> bool:
        """
        Registrar callback para un canal específico.
        
        Args:
            channel: Canal (ej: "ticker:BTC/USDT")
            callback: Función asincrónica a llamar con datos
            
        Returns:
            True si se registró correctamente
        """
        self.logger.info(f"Registrando callback para canal {channel}")
        
        self.callbacks[channel] = callback
        
        # Si el canal no está en suscripciones, suscribirse
        if channel not in self.subscriptions:
            await self.subscribe([channel])
            
        return True
        
    async def unregister_callback(self, channel: str) -> bool:
        """
        Eliminar callback para un canal.
        
        Args:
            channel: Canal
            
        Returns:
            True si se eliminó correctamente
        """
        self.logger.info(f"Eliminando callback para canal {channel}")
        
        if channel in self.callbacks:
            self.callbacks.pop(channel)
            
            # Si no hay más callbacks para este canal, cancelar suscripción
            await self.unsubscribe([channel])
            
        return True
        
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado actual del adaptador.
        
        Returns:
            Dict con estado actual
        """
        return {
            "exchange_id": self.exchange_id,
            "initialized": self.initialized,
            "connected": self.is_connected,
            "uptime": time.time() - self.start_time,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "subscription_count": len(self.subscriptions),
            "symbol_count": len(self.symbols)
        }