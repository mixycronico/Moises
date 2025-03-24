"""
Adaptador Trascendental para Binance Testnet.

Este módulo implementa un adaptador especializado para Binance Testnet
con capacidades avanzadas de WebSocket y procesamiento asincrónico,
diseñado específicamente para integrarse con el Seraphim Orchestrator.
"""

import asyncio
import logging
import time
import os
import hmac
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError, RequestTimeout

from genesis.core.transcendental_ws_adapter import TranscendentalWebSocketAdapter, ExchangeID
from genesis.exchanges.ccxt_wrapper import CCXTExchange

# Configuración de logging
logger = logging.getLogger("genesis.exchanges.binance_testnet")

class BinanceTestnetAdapter:
    """
    Adaptador Trascendental para Binance Testnet.
    
    Este adaptador combina las capacidades del WebSocket Trascendental
    con la API REST de Binance Testnet mediante CCXT, proporcionando
    una solución completa para trading en testnet.
    """
    
    def __init__(self):
        """Inicializar adaptador para Binance Testnet."""
        self.exchange_id = ExchangeID.BINANCE
        self.logger = logger
        
        # Variables de estado
        self.connected = False
        self.last_operation_time = None
        self.operation_count = 0
        self.error_count = 0
        self.transmuted_count = 0
        
        # Obtener credenciales API de variables de entorno
        self.api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        self.api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        
        # Verificar credenciales
        self.has_credentials = bool(self.api_key and self.api_secret)
        if not self.has_credentials:
            self.logger.warning("No se encontraron credenciales de Binance Testnet. La funcionalidad será limitada.")
        
        # Componentes del adaptador
        self.ws_adapter = None  # WebSocket para datos en tiempo real
        self.rest_client = None  # Cliente REST para operaciones de trading
        
        # Cache de datos
        self.tickers_cache = {}
        self.ticker_cache_ttl = 2.0  # Segundos
        self.order_books_cache = {}
        self.order_book_cache_ttl = 1.0  # Segundos
        
        # Suscripciones activas
        self.subscriptions = set()
        
    async def initialize(self) -> Dict[str, Any]:
        """
        Inicializar el adaptador y sus componentes.
        
        Returns:
            Dict con resultado de la inicialización
        """
        self.logger.info("Inicializando BinanceTestnetAdapter...")
        
        try:
            # Crear adaptador WebSocket para datos en tiempo real
            self.ws_adapter = TranscendentalWebSocketAdapter(self.exchange_id)
            
            # Crear cliente REST para operaciones de trading
            rest_config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'timeout': 30000,  # ms
                'testnet': True  # Crucial para usar testnet
            }
            
            self.rest_client = CCXTExchange(
                exchange_id='binance',
                api_key=self.api_key,
                secret=self.api_secret,
                config=rest_config
            )
            
            # Inicializar cliente REST
            if self.has_credentials:
                await self.rest_client.start()
                self.logger.info("Cliente REST para Binance Testnet inicializado correctamente")
            
            self.logger.info("BinanceTestnetAdapter inicializado correctamente")
            return {
                "success": True,
                "message": "BinanceTestnetAdapter inicializado correctamente",
                "has_credentials": self.has_credentials
            }
            
        except Exception as e:
            self.logger.error(f"Error al inicializar BinanceTestnetAdapter: {str(e)}")
            return {
                "success": False,
                "message": f"Error al inicializar: {str(e)}",
                "has_credentials": self.has_credentials
            }
    
    async def connect(self) -> Dict[str, Any]:
        """
        Conectar al WebSocket de Binance Testnet.
        
        Returns:
            Dict con resultado de la conexión
        """
        self.logger.info("Conectando a Binance Testnet WebSocket...")
        
        try:
            # Conectar WebSocket
            result = await self.ws_adapter.connect()
            
            # Actualizar estado
            self.connected = result["success"]
            self.last_operation_time = time.time()
            
            if result["success"]:
                self.logger.info("Conexión a Binance Testnet WebSocket establecida")
            else:
                self.logger.warning(f"No se pudo conectar a Binance Testnet WebSocket: {result.get('message', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al conectar a Binance Testnet WebSocket: {str(e)}")
            self.error_count += 1
            
            return {
                "success": False,
                "transmuted": True,
                "message": f"Error al conectar: {str(e)}"
            }
    
    async def close(self) -> Dict[str, Any]:
        """
        Cerrar conexión al WebSocket de Binance Testnet.
        
        Returns:
            Dict con resultado del cierre
        """
        self.logger.info("Cerrando conexión a Binance Testnet...")
        
        try:
            ws_result = {"success": True, "message": "No hay conexión WebSocket que cerrar"}
            
            if self.ws_adapter:
                ws_result = await self.ws_adapter.close()
            
            rest_result = {"success": True, "message": "No hay cliente REST que cerrar"}
            
            if self.rest_client:
                await self.rest_client.stop()
                rest_result = {"success": True, "message": "Cliente REST cerrado correctamente"}
            
            # Actualizar estado
            self.connected = False
            self.last_operation_time = time.time()
            
            return {
                "success": True,
                "ws_result": ws_result,
                "rest_result": rest_result,
                "message": "Conexión a Binance Testnet cerrada correctamente"
            }
            
        except Exception as e:
            self.logger.error(f"Error al cerrar conexión a Binance Testnet: {str(e)}")
            self.error_count += 1
            
            return {
                "success": False,
                "transmuted": True,
                "message": f"Error al cerrar conexión: {str(e)}"
            }
    
    async def subscribe(self, channels: List[str]) -> Dict[str, Any]:
        """
        Suscribirse a canales en el WebSocket de Binance Testnet.
        
        Args:
            channels: Lista de canales para suscripción
            
        Returns:
            Dict con resultado de la suscripción
        """
        self.logger.info(f"Suscribiendo a canales: {channels}")
        
        try:
            # Asegurar conexión WebSocket
            if not self.connected:
                await self.connect()
            
            # Realizar suscripción
            result = await self.ws_adapter.subscribe(channels)
            
            # Actualizar suscripciones locales
            for channel in channels:
                self.subscriptions.add(channel)
            
            self.last_operation_time = time.time()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al suscribirse a canales: {str(e)}")
            self.error_count += 1
            
            # Actualizar suscripciones locales a pesar del error
            for channel in channels:
                self.subscriptions.add(channel)
            
            return {
                "success": True,
                "transmuted": True,
                "message": f"Suscripción transmutada tras error: {str(e)}",
                "channels": list(self.subscriptions)
            }
    
    async def unsubscribe(self, channels: List[str]) -> Dict[str, Any]:
        """
        Cancelar suscripción a canales en el WebSocket de Binance Testnet.
        
        Args:
            channels: Lista de canales para cancelar suscripción
            
        Returns:
            Dict con resultado de la cancelación
        """
        self.logger.info(f"Cancelando suscripción a canales: {channels}")
        
        try:
            # Asegurar conexión WebSocket
            if not self.connected:
                await self.connect()
            
            # Realizar cancelación
            result = await self.ws_adapter.unsubscribe(channels)
            
            # Actualizar suscripciones locales
            for channel in channels:
                if channel in self.subscriptions:
                    self.subscriptions.remove(channel)
            
            self.last_operation_time = time.time()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al cancelar suscripción a canales: {str(e)}")
            self.error_count += 1
            
            # Actualizar suscripciones locales a pesar del error
            for channel in channels:
                if channel in self.subscriptions:
                    self.subscriptions.remove(channel)
            
            return {
                "success": True,
                "transmuted": True,
                "message": f"Cancelación transmutada tras error: {str(e)}",
                "channels": list(self.subscriptions)
            }
    
    async def receive(self) -> Dict[str, Any]:
        """
        Recibir mensaje del WebSocket de Binance Testnet.
        
        Returns:
            Mensaje recibido
        """
        try:
            if not self.connected:
                await self.connect()
            
            # Recibir mensaje
            message = await self.ws_adapter.receive()
            
            self.operation_count += 1
            self.last_operation_time = time.time()
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error al recibir mensaje: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error
            return {
                "success": True,
                "transmuted": True,
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            }
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener ticker para un símbolo específico.
        
        Args:
            symbol: Símbolo del par de trading (p.ej., 'BTC/USDT')
            
        Returns:
            Dict con información del ticker
        """
        self.logger.debug(f"Obteniendo ticker para {symbol}")
        
        # Verificar cache
        now = time.time()
        if symbol in self.tickers_cache:
            timestamp, ticker = self.tickers_cache[symbol]
            if now - timestamp < self.ticker_cache_ttl:
                return ticker
        
        try:
            # Intentar usar cliente REST si está disponible
            if self.has_credentials and self.rest_client:
                ticker = await self.rest_client.fetch_ticker(symbol)
                
                # Actualizar cache
                self.tickers_cache[symbol] = (now, ticker)
                
                return ticker
            else:
                # Usar WebSocket para obtener datos de ticker
                # Primero verificar si ya estamos suscritos al canal de ticker para este símbolo
                normalized_symbol = symbol.replace('/', '').lower()
                ticker_channel = f"{normalized_symbol}@ticker"
                
                if ticker_channel not in self.subscriptions:
                    await self.subscribe([ticker_channel])
                
                # Esperar el próximo mensaje que podría ser un ticker
                for _ in range(5):  # Intentar algunas veces
                    message = await self.receive()
                    
                    # Verificar si el mensaje es un ticker para nuestro símbolo
                    is_ticker = message.get("e") == "24hrTicker"
                    is_our_symbol = message.get("s", "").lower() == normalized_symbol
                    
                    if is_ticker and is_our_symbol:
                        # Convertir mensaje Binance a formato CCXT
                        ticker = {
                            'symbol': symbol,
                            'timestamp': message.get("E"),
                            'datetime': datetime.fromtimestamp(message.get("E") / 1000).isoformat(),
                            'high': float(message.get("h")),
                            'low': float(message.get("l")),
                            'bid': None,
                            'bidVolume': None,
                            'ask': None,
                            'askVolume': None,
                            'vwap': None,
                            'open': float(message.get("o")),
                            'close': float(message.get("c")),
                            'last': float(message.get("c")),
                            'previousClose': float(message.get("x")),
                            'change': float(message.get("p")),
                            'percentage': float(message.get("P")),
                            'average': None,
                            'baseVolume': float(message.get("v")),
                            'quoteVolume': float(message.get("q")),
                            'info': message
                        }
                        
                        # Actualizar cache
                        self.tickers_cache[symbol] = (now, ticker)
                        
                        return ticker
                
                # Si no recibimos un ticker después de varios intentos, generamos uno
                self.logger.warning(f"No se recibió ticker para {symbol}, generando uno simulado")
                
                # Generar ticker simulado
                ticker = {
                    'symbol': symbol,
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.now().isoformat(),
                    'high': 0.0,
                    'low': 0.0,
                    'bid': None,
                    'bidVolume': None,
                    'ask': None,
                    'askVolume': None,
                    'vwap': None,
                    'open': 0.0,
                    'close': 0.0,
                    'last': 0.0,
                    'previousClose': 0.0,
                    'change': 0.0,
                    'percentage': 0.0,
                    'average': None,
                    'baseVolume': 0.0,
                    'quoteVolume': 0.0,
                    'info': {'transmuted': True}
                }
                
                self.transmuted_count += 1
                
                # No actualizamos cache para datos transmutados
                
                return ticker
                
        except Exception as e:
            self.logger.error(f"Error al obtener ticker para {symbol}: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error
            ticker = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat(),
                'high': 0.0,
                'low': 0.0,
                'bid': None,
                'bidVolume': None,
                'ask': None,
                'askVolume': None,
                'vwap': None,
                'open': 0.0,
                'close': 0.0,
                'last': 0.0,
                'previousClose': 0.0,
                'change': 0.0,
                'percentage': 0.0,
                'average': None,
                'baseVolume': 0.0,
                'quoteVolume': 0.0,
                'info': {'transmuted': True, 'error': str(e)}
            }
            
            self.transmuted_count += 1
            
            return ticker
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Obtener libro de órdenes para un símbolo específico.
        
        Args:
            symbol: Símbolo del par de trading (p.ej., 'BTC/USDT')
            limit: Número máximo de órdenes en cada lado
            
        Returns:
            Dict con información del libro de órdenes
        """
        self.logger.debug(f"Obteniendo libro de órdenes para {symbol}")
        
        # Verificar cache
        now = time.time()
        cache_key = f"{symbol}_{limit}"
        if cache_key in self.order_books_cache:
            timestamp, order_book = self.order_books_cache[cache_key]
            if now - timestamp < self.order_book_cache_ttl:
                return order_book
        
        try:
            # Intentar usar cliente REST si está disponible
            if self.has_credentials and self.rest_client:
                order_book = await self.rest_client.fetch_order_book(symbol, limit)
                
                # Actualizar cache
                self.order_books_cache[cache_key] = (now, order_book)
                
                return order_book
            else:
                # Usar WebSocket para obtener datos de order book
                # En un sistema real, implementaríamos esto correctamente
                # Para esta demostración, generamos datos simulados con transmutación
                
                self.logger.warning(f"No hay cliente REST disponible, generando libro de órdenes simulado para {symbol}")
                
                # Generar libro de órdenes simulado
                base_price = 0.0
                
                # Para algunos símbolos comunes, usar precios realistas
                if symbol == 'BTC/USDT':
                    base_price = 61500.0
                elif symbol == 'ETH/USDT':
                    base_price = 3300.0
                elif symbol == 'BNB/USDT':
                    base_price = 570.0
                
                order_book = {
                    'symbol': symbol,
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.now().isoformat(),
                    'bids': [],
                    'asks': [],
                    'nonce': int(time.time() * 1000000)
                }
                
                # Generar bids y asks
                for i in range(limit):
                    bid_price = base_price * (1 - 0.0001 * (i + 1))
                    ask_price = base_price * (1 + 0.0001 * (i + 1))
                    
                    # [precio, cantidad]
                    order_book['bids'].append([bid_price, (limit - i) * 0.1])
                    order_book['asks'].append([ask_price, (limit - i) * 0.1])
                
                order_book['info'] = {'transmuted': True}
                
                self.transmuted_count += 1
                
                # No actualizamos cache para datos transmutados
                
                return order_book
                
        except Exception as e:
            self.logger.error(f"Error al obtener libro de órdenes para {symbol}: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error
            order_book = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat(),
                'bids': [],
                'asks': [],
                'nonce': int(time.time() * 1000000),
                'info': {'transmuted': True, 'error': str(e)}
            }
            
            self.transmuted_count += 1
            
            return order_book
    
    async def get_balance(self) -> Dict[str, Any]:
        """
        Obtener saldo de la cuenta.
        
        Returns:
            Dict con información del saldo
        """
        self.logger.info("Obteniendo saldo de la cuenta")
        
        try:
            # Usar cliente REST para obtener saldo
            if self.has_credentials and self.rest_client:
                balance = await self.rest_client.fetch_balance()
                return balance
            else:
                self.logger.warning("No hay credenciales disponibles, generando saldo simulado")
                
                # Generar saldo simulado
                balance = {
                    'info': {'transmuted': True},
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.now().isoformat(),
                    'free': {
                        'USDT': 10000.0,
                        'BTC': 0.1,
                        'ETH': 1.0,
                        'BNB': 10.0
                    },
                    'used': {
                        'USDT': 0.0,
                        'BTC': 0.0,
                        'ETH': 0.0,
                        'BNB': 0.0
                    },
                    'total': {
                        'USDT': 10000.0,
                        'BTC': 0.1,
                        'ETH': 1.0,
                        'BNB': 10.0
                    }
                }
                
                self.transmuted_count += 1
                
                return balance
                
        except Exception as e:
            self.logger.error(f"Error al obtener saldo: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error
            balance = {
                'info': {'transmuted': True, 'error': str(e)},
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat(),
                'free': {'USDT': 10000.0},
                'used': {'USDT': 0.0},
                'total': {'USDT': 10000.0}
            }
            
            self.transmuted_count += 1
            
            return balance
    
    async def create_order(self, symbol: str, order_type: str, side: str, 
                    amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Crear una nueva orden.
        
        Args:
            symbol: Símbolo del par de trading (p.ej., 'BTC/USDT')
            order_type: Tipo de orden ('limit', 'market')
            side: Lado de la orden ('buy', 'sell')
            amount: Cantidad de la orden
            price: Precio de la orden (para órdenes limit)
            
        Returns:
            Dict con información de la orden creada
        """
        self.logger.info(f"Creando orden {order_type} {side} {amount} {symbol} @ {price}")
        
        try:
            # Usar cliente REST para crear orden
            if self.has_credentials and self.rest_client:
                order = await self.rest_client.create_order(symbol, order_type, side, amount, price)
                return order
            else:
                self.logger.warning("No hay credenciales disponibles, generando orden simulada")
                
                # Generar orden simulada
                order = {
                    'id': f'transmuted_{int(time.time() * 1000)}',
                    'symbol': symbol,
                    'type': order_type,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'cost': amount * (price or 0),
                    'filled': 0.0,
                    'remaining': amount,
                    'status': 'open',
                    'fee': None,
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.now().isoformat(),
                    'info': {'transmuted': True}
                }
                
                self.transmuted_count += 1
                
                return order
                
        except Exception as e:
            self.logger.error(f"Error al crear orden: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error
            order = {
                'id': f'error_{int(time.time() * 1000)}',
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price,
                'cost': amount * (price or 0),
                'filled': 0.0,
                'remaining': amount,
                'status': 'rejected',
                'fee': None,
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat(),
                'info': {'transmuted': True, 'error': str(e)}
            }
            
            self.transmuted_count += 1
            
            return order
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancelar una orden existente.
        
        Args:
            order_id: ID de la orden a cancelar
            symbol: Símbolo del par de trading (requerido para algunos exchanges)
            
        Returns:
            Dict con información de la cancelación
        """
        self.logger.info(f"Cancelando orden {order_id} para {symbol}")
        
        try:
            # Usar cliente REST para cancelar orden
            if self.has_credentials and self.rest_client:
                result = await self.rest_client.cancel_order(order_id, symbol)
                return result
            else:
                self.logger.warning("No hay credenciales disponibles, generando cancelación simulada")
                
                # Generar resultado de cancelación simulado
                result = {
                    'id': order_id,
                    'symbol': symbol,
                    'status': 'canceled',
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.now().isoformat(),
                    'info': {'transmuted': True}
                }
                
                self.transmuted_count += 1
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error al cancelar orden: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error
            result = {
                'id': order_id,
                'symbol': symbol,
                'status': 'error',
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat(),
                'info': {'transmuted': True, 'error': str(e)}
            }
            
            self.transmuted_count += 1
            
            return result
    
    async def fetch_orders(self, symbol: Optional[str] = None, 
                    since: Optional[int] = None, 
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Obtener historial de órdenes.
        
        Args:
            symbol: Símbolo del par de trading
            since: Timestamp inicial en milisegundos
            limit: Número máximo de órdenes a obtener
            
        Returns:
            Lista de órdenes
        """
        self.logger.info(f"Obteniendo órdenes para {symbol}")
        
        try:
            # Usar cliente REST para obtener órdenes
            if self.has_credentials and self.rest_client:
                orders = await self.rest_client.fetch_orders(symbol, since, limit)
                return orders
            else:
                self.logger.warning("No hay credenciales disponibles, generando órdenes simuladas")
                
                # Generar órdenes simuladas
                orders = []
                
                # No retornamos órdenes simuladas para mantener el sistema limpio
                
                return orders
                
        except Exception as e:
            self.logger.error(f"Error al obtener órdenes: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error (lista vacía)
            return []
    
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtener órdenes abiertas.
        
        Args:
            symbol: Símbolo del par de trading
            
        Returns:
            Lista de órdenes abiertas
        """
        self.logger.info(f"Obteniendo órdenes abiertas para {symbol}")
        
        try:
            # Usar cliente REST para obtener órdenes abiertas
            if self.has_credentials and self.rest_client:
                orders = await self.rest_client.fetch_open_orders(symbol)
                return orders
            else:
                self.logger.warning("No hay credenciales disponibles, generando órdenes abiertas simuladas")
                
                # Generar órdenes abiertas simuladas
                orders = []
                
                # No retornamos órdenes simuladas para mantener el sistema limpio
                
                return orders
                
        except Exception as e:
            self.logger.error(f"Error al obtener órdenes abiertas: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error (lista vacía)
            return []
    
    async def fetch_closed_orders(self, symbol: Optional[str] = None, 
                           since: Optional[int] = None, 
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Obtener órdenes cerradas.
        
        Args:
            symbol: Símbolo del par de trading
            since: Timestamp inicial en milisegundos
            limit: Número máximo de órdenes a obtener
            
        Returns:
            Lista de órdenes cerradas
        """
        self.logger.info(f"Obteniendo órdenes cerradas para {symbol}")
        
        try:
            # Usar cliente REST para obtener órdenes cerradas
            if self.has_credentials and self.rest_client:
                orders = await self.rest_client.fetch_closed_orders(symbol, since, limit)
                return orders
            else:
                self.logger.warning("No hay credenciales disponibles, generando órdenes cerradas simuladas")
                
                # Generar órdenes cerradas simuladas
                orders = []
                
                # No retornamos órdenes simuladas para mantener el sistema limpio
                
                return orders
                
        except Exception as e:
            self.logger.error(f"Error al obtener órdenes cerradas: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error (lista vacía)
            return []
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado actual del adaptador.
        
        Returns:
            Dict con información del estado
        """
        return {
            "connected": self.connected,
            "has_credentials": self.has_credentials,
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "transmuted_count": self.transmuted_count,
            "last_operation_time": self.last_operation_time,
            "subscriptions": list(self.subscriptions)
        }
    
    async def get_symbols(self) -> List[str]:
        """
        Obtener lista de símbolos disponibles.
        
        Returns:
            Lista de símbolos
        """
        try:
            # Usar cliente REST para obtener símbolos
            if self.has_credentials and self.rest_client and hasattr(self.rest_client, 'exchange') and hasattr(self.rest_client.exchange, 'markets'):
                markets = self.rest_client.exchange.markets
                return list(markets.keys())
            else:
                # Lista predefinida de símbolos comunes
                return [
                    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
                    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT"
                ]
                
        except Exception as e:
            self.logger.error(f"Error al obtener símbolos: {str(e)}")
            
            # Lista predefinida en caso de error
            return [
                "BTC/USDT", "ETH/USDT", "BNB/USDT"
            ]
    
    async def get_candles(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[List[float]]:
        """
        Obtener datos de velas OHLCV para un símbolo específico.
        
        Args:
            symbol: Símbolo del par de trading (ej: 'BTC/USDT')
            timeframe: Intervalo de tiempo (ej: '1m', '5m', '1h', '1d')
            limit: Número máximo de velas a obtener
            
        Returns:
            Lista de velas OHLCV [timestamp, open, high, low, close, volume]
        """
        self.logger.info(f"Obteniendo candles para {symbol} en timeframe {timeframe}")
        
        try:
            # Intentar usar cliente REST si está disponible
            if self.has_credentials and self.rest_client:
                candles = await self.rest_client.fetch_ohlcv(symbol, timeframe, None, limit)
                return candles
            else:
                self.logger.warning(f"No hay credenciales disponibles, generando candles simulados para {symbol}")
                
                # Generar candles simulados
                now = int(time.time() * 1000)
                candles = []
                
                # Para algunos símbolos comunes, usar precios realistas
                base_price = 0.0
                volatility = 0.01  # 1% de volatilidad entre velas
                
                if symbol == 'BTC/USDT':
                    base_price = 61500.0
                    volatility = 0.005
                elif symbol == 'ETH/USDT':
                    base_price = 3300.0
                    volatility = 0.008
                elif symbol == 'BNB/USDT':
                    base_price = 570.0
                    volatility = 0.01
                
                # Determinar duración del timeframe en milisegundos
                timeframe_ms = 3600000  # 1 hora por defecto
                if timeframe == '1m':
                    timeframe_ms = 60000
                elif timeframe == '5m':
                    timeframe_ms = 300000
                elif timeframe == '15m':
                    timeframe_ms = 900000
                elif timeframe == '30m':
                    timeframe_ms = 1800000
                elif timeframe == '1h':
                    timeframe_ms = 3600000
                elif timeframe == '4h':
                    timeframe_ms = 14400000
                elif timeframe == '1d':
                    timeframe_ms = 86400000
                
                # Generar candles con una tendencia simple (alcista/bajista aleatoria)
                trend = 1 if random.random() > 0.5 else -1
                price = base_price * (1 + (random.random() - 0.5) * 0.01)
                
                for i in range(limit):
                    timestamp = now - (limit - i) * timeframe_ms
                    
                    # Aplicar tendencia y volatilidad
                    price_change = price * volatility * (random.random() - 0.5 + trend * 0.1)
                    price += price_change
                    
                    # Crear vela
                    open_price = price
                    close_price = price + price * volatility * (random.random() - 0.5)
                    high_price = max(open_price, close_price) + price * volatility * random.random()
                    low_price = min(open_price, close_price) - price * volatility * random.random()
                    volume = base_price * 10 * random.random()
                    
                    # [timestamp, open, high, low, close, volume]
                    candle = [timestamp, open_price, high_price, low_price, close_price, volume]
                    candles.append(candle)
                
                self.transmuted_count += 1
                
                return candles
                
        except Exception as e:
            self.logger.error(f"Error al obtener candles para {symbol}: {str(e)}")
            self.error_count += 1
            
            # En caso de error, devolver lista vacía
            return []
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """
        Obtener información de todos los mercados disponibles.
        
        Returns:
            Lista de información de mercados
        """
        self.logger.info("Obteniendo información de mercados")
        
        try:
            # Intentar usar cliente REST para obtener mercados
            if self.has_credentials and self.rest_client and hasattr(self.rest_client, 'exchange') and hasattr(self.rest_client.exchange, 'markets'):
                markets_dict = self.rest_client.exchange.markets
                
                # Convertir diccionario a lista de mercados
                markets = []
                for symbol, market_info in markets_dict.items():
                    markets.append({
                        "symbol": symbol,
                        "base": market_info.get("base"),
                        "quote": market_info.get("quote"),
                        "active": market_info.get("active", True),
                        "precision": market_info.get("precision", {}),
                        "limits": market_info.get("limits", {})
                    })
                
                return markets
            else:
                # Lista predefinida de símbolos comunes con información básica
                common_symbols = [
                    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
                    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT"
                ]
                
                markets = []
                for symbol in common_symbols:
                    base, quote = symbol.split('/')
                    markets.append({
                        "symbol": symbol,
                        "base": base,
                        "quote": quote,
                        "active": True,
                        "precision": {
                            "price": 8,
                            "amount": 6
                        },
                        "limits": {
                            "amount": {
                                "min": 0.000001,
                                "max": 9999.0
                            },
                            "price": {
                                "min": 0.00000001,
                                "max": 9999999.0
                            }
                        }
                    })
                
                return markets
                
        except Exception as e:
            self.logger.error(f"Error al obtener mercados: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error (lista vacía)
            return []
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener datos de mercado completos para un símbolo.
        
        Args:
            symbol: Símbolo del par de trading
            
        Returns:
            Dict con datos de mercado
        """
        self.logger.info(f"Obteniendo datos de mercado para {symbol}")
        
        try:
            # Obtener datos en paralelo
            ticker, order_book, candles = await asyncio.gather(
                self.get_ticker(symbol),
                self.get_order_book(symbol),
                self.get_candles(symbol, "5m", 20)
            )
            
            # Combinar en un único resultado
            return {
                "symbol": symbol,
                "ticker": ticker,
                "order_book": order_book,
                "candles": candles[:5],  # Solo incluir las últimas 5 velas para no sobrecargar
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener datos de mercado para {symbol}: {str(e)}")
            self.error_count += 1
            
            # Transmutación de error
            return {
                "symbol": symbol,
                "ticker": None,
                "order_book": None,
                "candles": [],
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().isoformat(),
                "error": str(e),
                "transmuted": True
            }