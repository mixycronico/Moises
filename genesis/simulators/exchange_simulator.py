"""
Simulador Ultra-Divino para intercambios de criptomonedas.

Este módulo implementa un simulador de intercambio completo con capacidades de:
1. Generación de datos de mercado realistas
2. Simulación de comportamiento de mercado con factores emocionales
3. Reacción a eventos externos
4. Procesamiento de órdenes con profundidad de mercado
5. Latencia variable configurable
6. Comportamiento de falla controlable
"""

import asyncio
import logging
import json
import math
import random
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Coroutine, Union

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Genesis.Simulator")

class MarketPattern(Enum):
    """Patrones de comportamiento de mercado simulado."""
    RANDOM_WALK = auto()          # Movimiento aleatorio (paseo aleatorio)
    TRENDING_UP = auto()          # Tendencia alcista 
    TRENDING_DOWN = auto()        # Tendencia bajista
    VOLATILE = auto()             # Alta volatilidad
    CONSOLIDATION = auto()        # Consolidación en rango
    BREAKOUT = auto()             # Ruptura de patrones
    LIQUIDITY_ZONE = auto()       # Zona de liquidez (concentración de órdenes)
    FLASH_CRASH = auto()          # Caída repentina
    FLASH_PUMP = auto()           # Alza repentina
    NEWS_REACTION = auto()        # Reacción a noticias
    EMOTIONAL_CYCLE = auto()      # Ciclo emocional (miedo y codicia)

class MarketEventType(Enum):
    """Tipos de eventos de mercado simulados."""
    POSITIVE_NEWS = auto()        # Noticias positivas
    NEGATIVE_NEWS = auto()        # Noticias negativas
    WHALE_BUY = auto()            # Compra de ballena
    WHALE_SELL = auto()           # Venta de ballena
    LIQUIDITY_SPIKE = auto()      # Aumento repentino de liquidez
    LIQUIDITY_DRAIN = auto()      # Disminución repentina de liquidez
    VOLATILITY_INCREASE = auto()  # Aumento de volatilidad
    VOLATILITY_DECREASE = auto()  # Disminución de volatilidad
    MARKET_OPEN = auto()          # Apertura de mercado
    MARKET_CLOSE = auto()         # Cierre de mercado
    TECHNICAL_BREAKDOWN = auto()  # Problemas técnicos

class HumanEmotionFactor(Enum):
    """Factores emocionales que influyen en el mercado simulado."""
    FEAR = auto()                 # Miedo
    GREED = auto()                # Codicia
    FOMO = auto()                 # Miedo a perderse algo (Fear Of Missing Out)
    PANIC = auto()                # Pánico
    EUPHORIA = auto()             # Euforia
    DISBELIEF = auto()            # Incredulidad
    HOPE = auto()                 # Esperanza
    CAPITULATION = auto()         # Capitulación
    COMPLACENCY = auto()          # Complacencia
    ANXIETY = auto()              # Ansiedad

class OrderType(Enum):
    """Tipos de órdenes soportados en el simulador."""
    MARKET = auto()               # Orden a mercado
    LIMIT = auto()                # Orden limitada
    STOP = auto()                 # Orden stop
    STOP_LIMIT = auto()           # Orden stop limitada
    TAKE_PROFIT = auto()          # Toma de beneficios
    TAKE_PROFIT_LIMIT = auto()    # Toma de beneficios limitada
    TRAILING_STOP = auto()        # Stop móvil

class OrderSide(Enum):
    """Lados de orden (compra/venta)."""
    BUY = auto()                  # Compra
    SELL = auto()                 # Venta

class OrderStatus(Enum):
    """Estados posibles de una orden."""
    NEW = auto()                  # Nueva orden
    PARTIALLY_FILLED = auto()     # Parcialmente ejecutada
    FILLED = auto()               # Completamente ejecutada
    CANCELED = auto()             # Cancelada
    REJECTED = auto()             # Rechazada
    EXPIRED = auto()              # Expirada
    
class ExchangeSimulator:
    """
    Simulador de intercambio de criptomonedas.
    
    Este simulador imita el comportamiento completo de un exchange,
    generando datos de mercado realistas y procesando órdenes.
    """
    
    def __init__(self, exchange_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar simulador de exchange.
        
        Args:
            exchange_id: Identificador del exchange simulado
            config: Configuración opcional del simulador
        """
        self.exchange_id = exchange_id
        self.logger = logger.getChild(f"Exchange.{exchange_id}")
        self.config = config or {}
        
        # Configuración predeterminada
        self.default_config = {
            "base_latency_ms": 50,         # Latencia base en ms
            "latency_variance": 0.2,       # Varianza de latencia (0-1)
            "error_rate": 0.01,            # Tasa de errores (0-1)
            "tick_interval_ms": 100,       # Intervalo entre actualizaciones de precio
            "default_spread": 0.0005,      # Spread predeterminado (0.05%)
            "min_order_size": 0.0001,      # Tamaño mínimo de orden
            "fee_rate": 0.001,             # Comisión base (0.1%)
            "liquidity_depth": 100,        # Profundidad de liquidez simulada
            "price_precision": 2,          # Precisión de precio (decimales)
            "quantity_precision": 8,       # Precisión de cantidad (decimales)
            "default_volume": 1000000,     # Volumen base simulado
            "volatility_factor": 0.02,     # Factor de volatilidad base (2%)
            "pattern_duration": 3600,      # Duración promedio de patrones (segundos)
            "emotional_cycle_period": 24,  # Período del ciclo emocional (horas)
            "news_impact_factor": 0.05,    # Impacto de noticias en precio (5%)
            "whale_size_factor": 0.2,      # Tamaño relativo de una ballena (20% del volumen)
            "failure_recovery_time": 5,    # Tiempo de recuperación tras fallo (segundos)
            "enable_failures": True,       # Habilitar fallos simulados
            "default_candle_count": 1000,  # Número de velas históricas generadas
            "default_timeframe": "1m",     # Timeframe predeterminado
            "enable_websocket": True,      # Habilitar simulación de websocket
        }
        
        # Combinar configuración predeterminada con personalizada
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Estado interno
        self.running = False
        self.websocket_running = False
        self.is_connected = False
        self.tickers = {}                   # symbol -> ticker_data
        self.orderbooks = {}                # symbol -> orderbook_data
        self.trades = {}                    # symbol -> recent_trades
        self.symbols = {}                   # symbol -> symbol_info
        self.orders = {}                    # order_id -> order_data
        self.positions = {}                 # symbol -> position_data
        self.candles = {}                   # symbol -> {timeframe -> candles}
        self.patterns = {}                  # symbol -> current_pattern
        self.emotions = {}                  # current_emotional_factors
        self.klines = {}                    # symbol -> {timeframe -> klines}
        self.listeners = {}                 # channel -> [callbacks]
        self.events = []                    # Lista de eventos programados
        self.ws_connections = set()         # Conexiones websocket activas
        self.errors = []                    # Registro de errores
        
        # Contadores y estadísticas
        self.tick_count = 0
        self.message_count = 0
        self.order_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.pattern_changes = 0
        self.event_count = 0
        
        # Instancias de mercado
        self.market_drivers = {}            # Controladores de mercado por símbolo
        
        # Tareas y bloqueos
        self.tasks = set()
        self.lock = asyncio.Lock()
        self.order_lock = asyncio.Lock()
        self.tick_task = None
        self.event_task = None
        
        self.logger.info(f"Simulador de exchange {exchange_id} inicializado")

    async def initialize(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Inicializar simulador con instrumentos de trading.
        
        Args:
            symbols: Lista opcional de símbolos para inicializar
                    (si es None, se usan símbolos predeterminados)
            
        Returns:
            Dict con estado de inicialización
        """
        default_symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", 
            "ADA/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT",
            "LINK/USDT", "UNI/USDT", "ATOM/USDT", "ALGO/USDT",
            "XRP/USDT", "LTC/USDT", "DOGE/USDT", "SHIB/USDT"
        ]
        
        # Usar símbolos proporcionados o predeterminados
        symbols_to_init = symbols if symbols is not None else default_symbols
        self.logger.info(f"Inicializando simulador con {len(symbols_to_init)} símbolos")
        
        # Inicializar símbolos
        for symbol in symbols_to_init:
            await self.add_symbol(symbol)
            
        self.running = True
        
        # Iniciar tareas de fondo
        self.tick_task = asyncio.create_task(self._ticker_updater())
        self.event_task = asyncio.create_task(self._event_processor())
        
        if self.config["enable_websocket"]:
            self.websocket_running = True
            
        self.is_connected = True
        
        return {
            "success": True,
            "message": f"Simulador inicializado con {len(symbols_to_init)} símbolos",
            "symbols": list(self.symbols.keys())
        }
        
    async def shutdown(self) -> Dict[str, Any]:
        """
        Detener el simulador y liberar recursos.
        
        Returns:
            Dict con estado de cierre
        """
        self.logger.info("Deteniendo simulador de exchange...")
        
        # Detener tareas de actualización
        self.running = False
        self.websocket_running = False
        
        # Cancelar todas las tareas
        if self.tick_task:
            self.tick_task.cancel()
            
        if self.event_task:
            self.event_task.cancel()
            
        for task in self.tasks:
            task.cancel()
            
        # Limpiar datos
        self.tickers.clear()
        self.orderbooks.clear()
        self.trades.clear()
        self.orders.clear()
        self.positions.clear()
        self.listeners.clear()
        self.ws_connections.clear()
        
        self.is_connected = False
        
        self.logger.info("Simulador de exchange detenido")
        return {
            "success": True,
            "message": "Simulador detenido correctamente"
        }
    
    async def add_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Añadir un nuevo símbolo/instrumento al simulador.
        
        Args:
            symbol: Símbolo a añadir (ej: "BTC/USDT")
            
        Returns:
            Dict con información del símbolo añadido
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        if symbol in self.symbols:
            return self.symbols[symbol]
        
        self.logger.info(f"Añadiendo símbolo: {symbol}")
        
        # Generar precios iniciales basados en típicos valores de mercado
        base_price = self._get_realistic_base_price(symbol)
        
        # Información de mercado para el símbolo
        symbol_data = {
            "symbol": symbol,
            "status": "TRADING",
            "baseAsset": symbol.split('/')[0],
            "quoteAsset": symbol.split('/')[1],
            "baseAssetPrecision": self.config["quantity_precision"],
            "quoteAssetPrecision": self.config["price_precision"],
            "filters": [
                {
                    "filterType": "PRICE_FILTER",
                    "minPrice": str(base_price / 1000),
                    "maxPrice": str(base_price * 1000),
                    "tickSize": str(10 ** -self.config["price_precision"])
                },
                {
                    "filterType": "LOT_SIZE",
                    "minQty": str(self.config["min_order_size"]),
                    "maxQty": "100000",
                    "stepSize": str(10 ** -self.config["quantity_precision"])
                }
            ],
            "permissions": ["SPOT", "MARGIN"],
            "createdTime": int(time.time() * 1000)
        }
        
        self.symbols[symbol] = symbol_data
        
        # Crear ticker inicial
        spread = self.config["default_spread"] * base_price
        self.tickers[symbol] = {
            "symbol": symbol,
            "price": base_price,
            "bid": base_price - spread/2,
            "ask": base_price + spread/2,
            "bidVolume": random.uniform(1, 10),
            "askVolume": random.uniform(1, 10),
            "high": base_price * 1.01,
            "low": base_price * 0.99,
            "volume": self.config["default_volume"],
            "quoteVolume": self.config["default_volume"] * base_price,
            "change": 0,
            "percentage": 0,
            "datetime": datetime.now().isoformat(),
            "timestamp": int(time.time() * 1000)
        }
        
        # Crear libro de órdenes inicial
        self.orderbooks[symbol] = self._generate_orderbook(symbol, base_price, self.config["liquidity_depth"])
        
        # Inicializar historial de operaciones
        self.trades[symbol] = []
        
        # Generar historial de velas
        self._generate_initial_candles(symbol, base_price)
        
        # Inicializar patrón de mercado aleatorio
        self.patterns[symbol] = random.choice(list(MarketPattern))
        
        # Crear controlador de mercado para este símbolo
        self.market_drivers[symbol] = self._create_market_driver(symbol, base_price)
        
        return symbol_data
    
    async def remove_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Eliminar un símbolo del simulador.
        
        Args:
            symbol: Símbolo a eliminar
            
        Returns:
            Dict con resultado de la operación
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        if symbol not in self.symbols:
            return {
                "success": False,
                "message": f"El símbolo {symbol} no existe"
            }
            
        self.logger.info(f"Eliminando símbolo: {symbol}")
        
        # Eliminar datos relacionados
        self.symbols.pop(symbol, None)
        self.tickers.pop(symbol, None)
        self.orderbooks.pop(symbol, None)
        self.trades.pop(symbol, None)
        self.patterns.pop(symbol, None)
        self.market_drivers.pop(symbol, None)
        
        # Eliminar velas
        if symbol in self.candles:
            self.candles.pop(symbol)
            
        if symbol in self.klines:
            self.klines.pop(symbol)
            
        # Eliminar posiciones
        positions_to_remove = [pos_id for pos_id, pos in self.positions.items() if pos["symbol"] == symbol]
        for pos_id in positions_to_remove:
            self.positions.pop(pos_id, None)
            
        # Cancelar órdenes
        orders_to_cancel = [order_id for order_id, order in self.orders.items() if order["symbol"] == symbol]
        for order_id in orders_to_cancel:
            self.orders.pop(order_id, None)
            
        return {
            "success": True,
            "message": f"Símbolo {symbol} eliminado correctamente"
        }
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener ticker para un símbolo.
        
        Args:
            symbol: Símbolo (ej: "BTC/USDT")
            
        Returns:
            Dict con datos de ticker
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        # Simulación de latencia
        await self._simulate_latency()
        
        # Simular error ocasional
        if random.random() < self.config["error_rate"] and self.config["enable_failures"]:
            self.error_count += 1
            error_msg = f"Error simulado al obtener ticker para {symbol}"
            self.errors.append({
                "timestamp": time.time(),
                "operation": "get_ticker",
                "symbol": symbol,
                "message": error_msg
            })
            self.logger.warning(error_msg)
            
            if random.random() < 0.5:  # 50% de probabilidad de recuperación transmutada
                await asyncio.sleep(self.config["failure_recovery_time"])
                # Usar último precio conocido con un ajuste aleatorio para la transmutación
                last_price = self.tickers.get(symbol, {"price": 1000})["price"]
                transmuted_price = last_price * random.uniform(0.99, 1.01)
                
                return {
                    "symbol": symbol,
                    "price": transmuted_price,
                    "transmuted": True,
                    "timestamp": int(time.time() * 1000)
                }
            else:
                # Error que no se recupera
                raise Exception(error_msg)
        
        if symbol not in self.tickers:
            # Si no existe, crear un ticker con datos ficticios
            await self.add_symbol(symbol)
            
        return self.tickers[symbol]
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Obtener libro de órdenes para un símbolo.
        
        Args:
            symbol: Símbolo (ej: "BTC/USDT")
            limit: Número máximo de niveles a devolver
            
        Returns:
            Dict con libro de órdenes
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        # Simulación de latencia
        await self._simulate_latency()
        
        # Simular error ocasional
        if random.random() < self.config["error_rate"] and self.config["enable_failures"]:
            self.error_count += 1
            error_msg = f"Error simulado al obtener orderbook para {symbol}"
            self.errors.append({
                "timestamp": time.time(),
                "operation": "get_orderbook",
                "symbol": symbol,
                "message": error_msg
            })
            self.logger.warning(error_msg)
            
            if random.random() < 0.5:  # 50% de probabilidad de recuperación transmutada
                await asyncio.sleep(self.config["failure_recovery_time"])
                
                # Crear un orderbook transmutado simple
                ticker = await self.get_ticker(symbol)
                price = ticker["price"]
                
                transmuted_orderbook = {
                    "symbol": symbol,
                    "bids": [[price * 0.99, 1.0]],
                    "asks": [[price * 1.01, 1.0]],
                    "transmuted": True,
                    "timestamp": int(time.time() * 1000)
                }
                
                return transmuted_orderbook
            else:
                # Error que no se recupera
                raise Exception(error_msg)
        
        if symbol not in self.orderbooks:
            # Si no existe, crear un símbolo y orderbook con datos ficticios
            await self.add_symbol(symbol)
            
        orderbook = self.orderbooks[symbol]
        
        # Limitar a la cantidad solicitada
        limited_orderbook = {
            "symbol": symbol,
            "bids": orderbook["bids"][:limit],
            "asks": orderbook["asks"][:limit],
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.now().isoformat(),
            "nonce": self.tick_count
        }
        
        return limited_orderbook
    
    async def get_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de operaciones recientes para un símbolo.
        
        Args:
            symbol: Símbolo (ej: "BTC/USDT")
            limit: Número máximo de operaciones a devolver
            
        Returns:
            Lista de operaciones recientes
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        # Simulación de latencia
        await self._simulate_latency()
        
        if symbol not in self.trades:
            # Si no existe, crear un símbolo y trades vacíos
            await self.add_symbol(symbol)
            
        # Si no hay suficientes trades, generar algunos
        if len(self.trades[symbol]) < limit:
            ticker = await self.get_ticker(symbol)
            base_price = ticker["price"]
            
            # Generar trades históricos
            current_time = time.time()
            for i in range(limit - len(self.trades[symbol])):
                # Generar hace 1-10 segundos, precio con variación pequeña
                trade_time = current_time - random.uniform(1, 10)
                price = base_price * random.uniform(0.998, 1.002)
                amount = random.uniform(0.01, 1.0)
                side = "buy" if random.random() > 0.5 else "sell"
                
                trade = {
                    "id": f"{self.exchange_id}-{symbol}-{int(trade_time * 1000)}",
                    "symbol": symbol,
                    "price": price,
                    "amount": amount,
                    "cost": price * amount,
                    "side": side,
                    "timestamp": int(trade_time * 1000),
                    "datetime": datetime.fromtimestamp(trade_time).isoformat()
                }
                
                self.trades[symbol].append(trade)
        
        # Ordenar por timestamp descendente y limitar
        recent_trades = sorted(self.trades[symbol], key=lambda x: x["timestamp"], reverse=True)[:limit]
        
        return recent_trades
    
    async def get_candles(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener velas OHLCV para un símbolo y timeframe.
        
        Args:
            symbol: Símbolo (ej: "BTC/USDT")
            timeframe: Período de tiempo (ej: "1m", "5m", "1h", "1d")
            limit: Número máximo de velas a devolver
            
        Returns:
            Lista de velas OHLCV
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        # Simulación de latencia
        await self._simulate_latency()
        
        if symbol not in self.candles or timeframe not in self.candles[symbol]:
            # Si no existe, inicializar símbolo y generar velas
            if symbol not in self.symbols:
                await self.add_symbol(symbol)
                
            # Inicializar diccionario de timeframes si es necesario
            if symbol not in self.candles:
                self.candles[symbol] = {}
                
            # Generar velas para este timeframe
            ticker = await self.get_ticker(symbol)
            base_price = ticker["price"]
            self._generate_candles_for_timeframe(symbol, timeframe, base_price)
            
        # Obtener las últimas 'limit' velas
        result = self.candles[symbol][timeframe][-limit:]
        
        return result
    
    async def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], Coroutine]) -> Dict[str, Any]:
        """
        Suscribirse a un canal de datos.
        
        Args:
            channel: Canal de suscripción
            callback: Función asincrónica a llamar con nuevos datos
            
        Returns:
            Dict con resultado de la suscripción
        """
        self.logger.info(f"Suscribiendo a canal: {channel}")
        
        # Añadir a la lista de listeners
        if channel not in self.listeners:
            self.listeners[channel] = []
            
        self.listeners[channel].append(callback)
        
        # Registrar conexión websocket
        connection_id = f"ws-{len(self.ws_connections) + 1}"
        self.ws_connections.add(connection_id)
        
        return {
            "success": True,
            "message": f"Suscrito a canal {channel}",
            "connection_id": connection_id,
            "timestamp": int(time.time() * 1000)
        }
    
    async def unsubscribe(self, channel: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Cancelar suscripción a un canal de datos.
        
        Args:
            channel: Canal de suscripción
            callback: Función específica a eliminar (None = todas)
            
        Returns:
            Dict con resultado de la cancelación
        """
        self.logger.info(f"Cancelando suscripción a canal: {channel}")
        
        if channel not in self.listeners:
            return {
                "success": False,
                "message": f"Canal {channel} no encontrado"
            }
            
        if callback is None:
            # Eliminar todos los callbacks para este canal
            self.listeners.pop(channel)
        else:
            # Eliminar solo este callback
            if callback in self.listeners[channel]:
                self.listeners[channel].remove(callback)
                
            # Si no quedan callbacks, eliminar el canal
            if not self.listeners[channel]:
                self.listeners.pop(channel)
                
        return {
            "success": True,
            "message": f"Suscripción a canal {channel} cancelada"
        }
    
    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Colocar una orden en el simulador.
        
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
        # Validar datos mínimos
        required_fields = ["symbol", "side", "type", "quantity"]
        for field in required_fields:
            if field not in order_data:
                return {
                    "success": False,
                    "message": f"Campo requerido ausente: {field}"
                }
                
        # Normalizar símbolo
        symbol = order_data["symbol"].upper()
        order_data["symbol"] = symbol
        
        # Validar que el símbolo existe
        if symbol not in self.symbols:
            await self.add_symbol(symbol)
            
        # Simulación de latencia
        await self._simulate_latency()
        
        # Simular error ocasional
        if random.random() < self.config["error_rate"] and self.config["enable_failures"]:
            self.error_count += 1
            error_msg = f"Error simulado al colocar orden para {symbol}"
            self.errors.append({
                "timestamp": time.time(),
                "operation": "place_order",
                "symbol": symbol,
                "message": error_msg
            })
            self.logger.warning(error_msg)
            
            if random.random() < 0.5:  # 50% de probabilidad de recuperación transmutada
                await asyncio.sleep(self.config["failure_recovery_time"])
                
                return {
                    "success": True,
                    "transmuted": True,
                    "message": "Orden transmutada tras error",
                    "order_id": f"transmuted-{int(time.time() * 1000)}",
                    "status": "NEW"
                }
            else:
                # Error que no se recupera
                raise Exception(error_msg)
        
        # Procesar la orden según su tipo
        async with self.order_lock:
            order_id = f"{self.exchange_id}-order-{int(time.time() * 1000)}-{self.order_count}"
            self.order_count += 1
            
            # Datos base de la orden
            order = {
                "id": order_id,
                "clientOrderId": order_data.get("clientOrderId", f"client-{order_id}"),
                "symbol": symbol,
                "side": order_data["side"],
                "type": order_data["type"],
                "quantity": float(order_data["quantity"]),
                "price": float(order_data.get("price", 0)),
                "status": OrderStatus.NEW.name,
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().isoformat(),
                "filled": 0,
                "remaining": float(order_data["quantity"]),
                "cost": 0,
                "trades": []
            }
            
            # Procesar según tipo de orden
            ticker = await self.get_ticker(symbol)
            current_price = ticker["price"]
            
            if order_data["type"].lower() == "market":
                # Orden a mercado (ejecución inmediata)
                order["price"] = current_price
                
                # Ejecutar totalmente la orden
                execution_price = current_price
                if order_data["side"].lower() == "buy":
                    # Ligero slippage en compras
                    execution_price *= random.uniform(1.0, 1.002)
                else:
                    # Ligero slippage en ventas
                    execution_price *= random.uniform(0.998, 1.0)
                    
                order["price"] = execution_price
                order["filled"] = order["quantity"]
                order["remaining"] = 0
                order["cost"] = order["quantity"] * execution_price
                order["status"] = OrderStatus.FILLED.name
                
                # Registrar la operación
                trade = {
                    "id": f"{self.exchange_id}-trade-{int(time.time() * 1000)}-{self.trade_count}",
                    "order": order_id,
                    "symbol": symbol,
                    "price": execution_price,
                    "amount": order["quantity"],
                    "cost": order["cost"],
                    "fee": order["cost"] * self.config["fee_rate"],
                    "side": order["side"],
                    "timestamp": int(time.time() * 1000),
                    "datetime": datetime.now().isoformat()
                }
                
                self.trade_count += 1
                order["trades"].append(trade)
                
                # Añadir al historial de operaciones
                self.trades[symbol].insert(0, trade)
                
                # Limitar historial de operaciones
                if len(self.trades[symbol]) > 1000:
                    self.trades[symbol] = self.trades[symbol][:1000]
                    
            elif order_data["type"].lower() == "limit":
                # Orden límite (queda en el libro hasta ejecutarse)
                if "price" not in order_data:
                    return {
                        "success": False,
                        "message": "Campo 'price' requerido para órdenes límite"
                    }
                    
                limit_price = float(order_data["price"])
                order["price"] = limit_price
                
                # Verificar si se ejecuta inmediatamente
                if (order_data["side"].lower() == "buy" and limit_price >= ticker["ask"]) or \
                   (order_data["side"].lower() == "sell" and limit_price <= ticker["bid"]):
                    # Ejecución inmediata
                    execution_price = limit_price
                    order["filled"] = order["quantity"]
                    order["remaining"] = 0
                    order["cost"] = order["quantity"] * execution_price
                    order["status"] = OrderStatus.FILLED.name
                    
                    # Registrar la operación
                    trade = {
                        "id": f"{self.exchange_id}-trade-{int(time.time() * 1000)}-{self.trade_count}",
                        "order": order_id,
                        "symbol": symbol,
                        "price": execution_price,
                        "amount": order["quantity"],
                        "cost": order["cost"],
                        "fee": order["cost"] * self.config["fee_rate"],
                        "side": order["side"],
                        "timestamp": int(time.time() * 1000),
                        "datetime": datetime.now().isoformat()
                    }
                    
                    self.trade_count += 1
                    order["trades"].append(trade)
                    
                    # Añadir al historial de operaciones
                    self.trades[symbol].insert(0, trade)
                    
                    # Limitar historial de operaciones
                    if len(self.trades[symbol]) > 1000:
                        self.trades[symbol] = self.trades[symbol][:1000]
                else:
                    # Queda pendiente en el libro
                    # Actualizar el libro de órdenes
                    self._add_order_to_orderbook(symbol, order)
            
            # Almacenar la orden
            self.orders[order_id] = order
            
            return {
                "success": True,
                "order": order
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancelar una orden existente.
        
        Args:
            order_id: ID de la orden a cancelar
            
        Returns:
            Dict con resultado de la operación
        """
        # Simulación de latencia
        await self._simulate_latency()
        
        if order_id not in self.orders:
            return {
                "success": False,
                "message": f"Orden {order_id} no encontrada"
            }
            
        order = self.orders[order_id]
        
        # Validar que la orden puede cancelarse
        if order["status"] in [OrderStatus.FILLED.name, OrderStatus.CANCELED.name]:
            return {
                "success": False,
                "message": f"No se puede cancelar orden en estado {order['status']}"
            }
            
        async with self.order_lock:
            # Actualizar estado
            order["status"] = OrderStatus.CANCELED.name
            order["timestamp"] = int(time.time() * 1000)
            order["datetime"] = datetime.now().isoformat()
            
            # Si estaba en el libro de órdenes, eliminarla
            symbol = order["symbol"]
            if symbol in self.orderbooks:
                self._remove_order_from_orderbook(symbol, order)
                
        return {
            "success": True,
            "order": order
        }
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Obtener información de una orden.
        
        Args:
            order_id: ID de la orden
            
        Returns:
            Dict con datos de la orden
        """
        # Simulación de latencia
        await self._simulate_latency()
        
        if order_id not in self.orders:
            return {
                "success": False,
                "message": f"Orden {order_id} no encontrada"
            }
            
        return {
            "success": True,
            "order": self.orders[order_id]
        }
    
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None, 
                        limit: int = 100) -> Dict[str, Any]:
        """
        Obtener órdenes filtradas por símbolo y/o estado.
        
        Args:
            symbol: Símbolo opcional para filtrar
            status: Estado opcional para filtrar
            limit: Número máximo de órdenes a devolver
            
        Returns:
            Dict con lista de órdenes
        """
        # Simulación de latencia
        await self._simulate_latency()
        
        # Filtrar órdenes
        filtered_orders = []
        
        for order_id, order in self.orders.items():
            # Aplicar filtros
            if symbol is not None and order["symbol"] != symbol.upper():
                continue
                
            if status is not None and order["status"] != status:
                continue
                
            filtered_orders.append(order)
            
        # Ordenar por timestamp descendente
        filtered_orders.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Limitar resultados
        limited_orders = filtered_orders[:limit]
        
        return {
            "success": True,
            "orders": limited_orders,
            "total": len(filtered_orders)
        }
    
    async def set_market_pattern(self, symbol: str, pattern: MarketPattern) -> Dict[str, Any]:
        """
        Establecer un patrón de mercado específico para un símbolo.
        
        Args:
            symbol: Símbolo para establecer patrón
            pattern: Patrón de mercado a establecer
            
        Returns:
            Dict con resultado de la operación
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        if symbol not in self.symbols:
            await self.add_symbol(symbol)
            
        self.logger.info(f"Estableciendo patrón {pattern.name} para {symbol}")
        self.patterns[symbol] = pattern
        
        # Actualizar driver de mercado
        if symbol in self.market_drivers:
            self.market_drivers[symbol]["pattern"] = pattern
            self.market_drivers[symbol]["pattern_start_time"] = time.time()
            self.market_drivers[symbol]["pattern_duration"] = random.uniform(
                0.5 * self.config["pattern_duration"],
                1.5 * self.config["pattern_duration"]
            )
            
        self.pattern_changes += 1
        
        return {
            "success": True,
            "message": f"Patrón {pattern.name} establecido para {symbol}",
            "previous_pattern": self.patterns[symbol].name if symbol in self.patterns else None
        }
    
    async def add_market_event(self, event_type: MarketEventType, symbol: str, 
                              impact: float = 0.02, delay: float = 0) -> Dict[str, Any]:
        """
        Añadir un evento de mercado programado.
        
        Args:
            event_type: Tipo de evento
            symbol: Símbolo afectado
            impact: Impacto del evento (0-1)
            delay: Retraso en segundos para activar el evento
            
        Returns:
            Dict con resultado de la operación
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        if symbol not in self.symbols:
            await self.add_symbol(symbol)
            
        event_time = time.time() + delay
        
        event = {
            "id": f"event-{len(self.events) + 1}",
            "type": event_type,
            "symbol": symbol,
            "impact": impact,
            "time": event_time,
            "executed": False
        }
        
        self.events.append(event)
        self.logger.info(f"Evento {event_type.name} programado para {symbol} en {delay}s con impacto {impact:.2%}")
        
        return {
            "success": True,
            "message": f"Evento {event_type.name} programado",
            "event_id": event["id"]
        }
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Obtener información general del exchange simulado.
        
        Returns:
            Dict con información del exchange
        """
        # Simulación de latencia
        await self._simulate_latency()
        
        return {
            "name": self.exchange_id,
            "symbols": list(self.symbols.keys()),
            "symbol_count": len(self.symbols),
            "order_count": self.order_count,
            "trade_count": self.trade_count,
            "uptime": time.time() - self.start_time,
            "current_time": int(time.time() * 1000),
            "server_time": int(time.time() * 1000),
            "limits": {
                "orders": {
                    "max": 100,
                    "per_minute": 50
                }
            },
            "version": "1.0.0",
            "status": {
                "status": "online",
                "message": "Simulador funcionando normalmente"
            }
        }
    
    async def set_ticker_price(self, symbol: str, price: float) -> Dict[str, Any]:
        """
        Establecer precio específico para un ticker (para pruebas).
        
        Args:
            symbol: Símbolo a modificar
            price: Nuevo precio
            
        Returns:
            Dict con resultado de la operación
        """
        symbol = symbol.upper()  # Normalizar símbolo
        
        if symbol not in self.symbols:
            await self.add_symbol(symbol)
            
        if symbol not in self.tickers:
            return {
                "success": False,
                "message": f"Ticker para {symbol} no encontrado"
            }
            
        old_price = self.tickers[symbol]["price"]
        
        # Actualizar ticker
        self.tickers[symbol]["price"] = price
        self.tickers[symbol]["bid"] = price * 0.999
        self.tickers[symbol]["ask"] = price * 1.001
        self.tickers[symbol]["change"] = price - old_price
        self.tickers[symbol]["percentage"] = ((price / old_price) - 1) * 100
        self.tickers[symbol]["timestamp"] = int(time.time() * 1000)
        self.tickers[symbol]["datetime"] = datetime.now().isoformat()
        
        # Si el nuevo precio es el máximo o mínimo del día, actualizar
        if price > self.tickers[symbol]["high"]:
            self.tickers[symbol]["high"] = price
            
        if price < self.tickers[symbol]["low"]:
            self.tickers[symbol]["low"] = price
            
        # Actualizar orderbook
        spread = price * self.config["default_spread"]
        self.orderbooks[symbol] = self._generate_orderbook(symbol, price, self.config["liquidity_depth"])
        
        # Notificar a subscribers
        await self._notify_ticker_update(symbol)
        
        return {
            "success": True,
            "message": f"Precio actualizado para {symbol}: {old_price} -> {price}"
        }
    
    # Métodos internos
    
    def _get_realistic_base_price(self, symbol: str) -> float:
        """
        Generar un precio base realista para un símbolo.
        
        Args:
            symbol: Símbolo para generar precio
            
        Returns:
            Precio base realista
        """
        # Simplifiquemos para facilidad de uso
        symbol = symbol.upper()
        
        # Precios típicos de referencia (actualizados a marzo 2023)
        typical_prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "BNB": 400.0,
            "SOL": 100.0,
            "ADA": 0.5,
            "DOT": 7.0,
            "AVAX": 30.0,
            "MATIC": 1.2,
            "LINK": 15.0,
            "UNI": 7.0,
            "ATOM": 10.0,
            "ALGO": 0.3,
            "XRP": 0.5,
            "LTC": 80.0,
            "DOGE": 0.1,
            "SHIB": 0.00002,
            "NEAR": 5.0,
            "FTM": 0.5,
            "SAND": 0.7,
            "MANA": 0.8
        }
        
        # Extraer token base
        parts = symbol.split('/')
        base_token = parts[0]
        
        # Usar precio típico si existe, o generar uno aleatorio
        if base_token in typical_prices:
            price = typical_prices[base_token]
            # Añadir pequeña variación
            price *= random.uniform(0.95, 1.05)
        else:
            # Para tokens desconocidos, generar un precio
            if random.random() < 0.1:
                # 10% de probabilidad de token de alto valor
                price = random.uniform(100, 1000)
            elif random.random() < 0.3:
                # 30% de probabilidad de token de valor medio
                price = random.uniform(1, 100)
            elif random.random() < 0.5:
                # 50% de probabilidad de token de bajo valor
                price = random.uniform(0.1, 1)
            else:
                # 10% de probabilidad de micro token
                price = random.uniform(0.00001, 0.1)
                
        return price
    
    def _generate_orderbook(self, symbol: str, base_price: float, depth: int) -> Dict[str, Any]:
        """
        Generar un libro de órdenes realista.
        
        Args:
            symbol: Símbolo
            base_price: Precio base
            depth: Profundidad del libro
            
        Returns:
            Libro de órdenes simulado
        """
        # Configurar los incrementos de precio
        base_increment = base_price * 0.0001  # 0.01% por nivel
        
        # Generar órdenes de compra (bids)
        bids = []
        current_bid = base_price * 0.999  # Empezar 0.1% por debajo del precio
        
        for i in range(depth):
            # Mayor volumen en niveles cercanos al precio
            volume = random.uniform(0.1, 5.0) * (depth - i) / depth
            bids.append([current_bid, volume])
            
            # Reducir el precio para el siguiente nivel
            increment = base_increment * (1 + random.uniform(-0.5, 0.5))
            current_bid -= increment
            
        # Generar órdenes de venta (asks)
        asks = []
        current_ask = base_price * 1.001  # Empezar 0.1% por encima del precio
        
        for i in range(depth):
            # Mayor volumen en niveles cercanos al precio
            volume = random.uniform(0.1, 5.0) * (depth - i) / depth
            asks.append([current_ask, volume])
            
            # Aumentar el precio para el siguiente nivel
            increment = base_increment * (1 + random.uniform(-0.5, 0.5))
            current_ask += increment
            
        # Estructura del libro de órdenes
        orderbook = {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.now().isoformat(),
            "nonce": self.tick_count
        }
        
        return orderbook
    
    def _generate_initial_candles(self, symbol: str, base_price: float) -> None:
        """
        Generar historial inicial de velas para un símbolo.
        
        Args:
            symbol: Símbolo
            base_price: Precio base
        """
        # Inicializar diccionario de timeframes
        if symbol not in self.candles:
            self.candles[symbol] = {}
            
        # Generar candles para timeframes comunes
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        for timeframe in timeframes:
            self._generate_candles_for_timeframe(symbol, timeframe, base_price)
            
    def _generate_candles_for_timeframe(self, symbol: str, timeframe: str, base_price: float) -> None:
        """
        Generar velas para un timeframe específico.
        
        Args:
            symbol: Símbolo
            timeframe: Timeframe
            base_price: Precio base
        """
        # Determinar duración de la vela en segundos
        if timeframe.endswith('m'):
            duration = int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            duration = int(timeframe[:-1]) * 60 * 60
        elif timeframe.endswith('d'):
            duration = int(timeframe[:-1]) * 24 * 60 * 60
        else:
            duration = 60  # 1 minuto por defecto
            
        # Determinar cantidad de velas a generar
        count = self.config["default_candle_count"]
        
        # Generar velas
        candles = []
        current_time = time.time()
        
        # Inicializar con precio base
        price = base_price
        
        # Patrones de mercado utilizados para generar datos históricos
        patterns = list(MarketPattern)
        current_pattern = random.choice(patterns)
        pattern_duration = 0
        pattern_max_duration = 20  # Velas con el mismo patrón
        
        # Generar velas del más antiguo al más reciente
        for i in range(count - 1, -1, -1):
            # Cambiar patrón ocasionalmente
            pattern_duration += 1
            if pattern_duration >= pattern_max_duration:
                current_pattern = random.choice(patterns)
                pattern_duration = 0
                pattern_max_duration = random.randint(10, 30)
            
            # Calcular tiempo de esta vela
            candle_time = current_time - (i + 1) * duration
            
            # Volatilidad base según timeframe
            if timeframe.endswith('m'):
                base_volatility = 0.002  # 0.2% para minutos
            elif timeframe.endswith('h'):
                base_volatility = 0.005  # 0.5% para horas
            else:
                base_volatility = 0.01   # 1% para días
                
            # Ajustar volatilidad según patrón
            if current_pattern == MarketPattern.VOLATILE:
                volatility = base_volatility * 3
            elif current_pattern == MarketPattern.CONSOLIDATION:
                volatility = base_volatility * 0.5
            else:
                volatility = base_volatility
                
            # Generar cambio de precio para esta vela
            if current_pattern == MarketPattern.TRENDING_UP:
                change = random.uniform(0, volatility * 2)
            elif current_pattern == MarketPattern.TRENDING_DOWN:
                change = random.uniform(-volatility * 2, 0)
            elif current_pattern == MarketPattern.RANDOM_WALK:
                change = random.uniform(-volatility, volatility)
            elif current_pattern == MarketPattern.FLASH_CRASH and random.random() < 0.2:
                change = -volatility * 5
            elif current_pattern == MarketPattern.FLASH_PUMP and random.random() < 0.2:
                change = volatility * 5
            else:
                change = random.uniform(-volatility, volatility)
                
            # Calcular precio con el cambio
            next_price = price * (1 + change)
            
            # Generar OHLC para la vela
            high = max(price, next_price) * (1 + random.uniform(0, volatility * 0.5))
            low = min(price, next_price) * (1 - random.uniform(0, volatility * 0.5))
            
            # Asegurar que high >= open/close >= low
            high = max(high, price, next_price)
            low = min(low, price, next_price)
            
            # Generar volumen
            if current_pattern in [MarketPattern.FLASH_CRASH, MarketPattern.FLASH_PUMP, 
                                MarketPattern.BREAKOUT, MarketPattern.VOLATILE]:
                # Mayor volumen en patrones extremos
                volume = self.config["default_volume"] * random.uniform(1.5, 3.0)
            else:
                volume = self.config["default_volume"] * random.uniform(0.5, 1.5)
                
            # Crear vela OHLCV
            candle = {
                "timestamp": int(candle_time * 1000),
                "datetime": datetime.fromtimestamp(candle_time).isoformat(),
                "open": price,
                "high": high,
                "low": low,
                "close": next_price,
                "volume": volume
            }
            
            candles.append(candle)
            
            # Preparar para la siguiente vela
            price = next_price
            
        # Almacenar las velas generadas
        self.candles[symbol][timeframe] = candles
    
    def _create_market_driver(self, symbol: str, base_price: float) -> Dict[str, Any]:
        """
        Crear un controlador de mercado para un símbolo.
        
        Args:
            symbol: Símbolo
            base_price: Precio base
            
        Returns:
            Dict con configuración del controlador
        """
        # Asignar un patrón inicial
        pattern = random.choice(list(MarketPattern))
        
        # Configuración del driver
        driver = {
            "symbol": symbol,
            "base_price": base_price,
            "current_price": base_price,
            "pattern": pattern,
            "pattern_start_time": time.time(),
            "pattern_duration": random.uniform(
                0.5 * self.config["pattern_duration"],
                1.5 * self.config["pattern_duration"]
            ),
            "emotion_factors": {
                emotion: random.uniform(0.1, 0.9) for emotion in HumanEmotionFactor
            },
            "last_update": time.time(),
            "volatility": self.config["volatility_factor"],
            "momentum": 0.0,
            "trend_strength": random.uniform(0.1, 0.9),
            "support_levels": [
                base_price * 0.95,
                base_price * 0.9,
                base_price * 0.85
            ],
            "resistance_levels": [
                base_price * 1.05,
                base_price * 1.1,
                base_price * 1.15
            ],
            "reversal_probability": 0.01,
            "event_impact_decay": 0.9,
            "active_events": []
        }
        
        return driver
    
    async def _ticker_updater(self) -> None:
        """Tarea de fondo para actualizar tickers y orderbooks."""
        self.logger.info("Iniciando tarea de actualización de tickers")
        
        while self.running:
            try:
                # Actualizar contador de ticks
                self.tick_count += 1
                
                # Para cada símbolo activo, actualizar precios
                for symbol in list(self.symbols.keys()):
                    try:
                        # Obtener driver de mercado
                        if symbol in self.market_drivers:
                            driver = self.market_drivers[symbol]
                            
                            # Actualizar precio según patrón y emociones
                            price = await self._update_market_price(symbol, driver)
                            
                            # Actualizar ticker
                            if symbol in self.tickers:
                                old_price = self.tickers[symbol]["price"]
                                
                                # Actualizar campos del ticker
                                self.tickers[symbol]["price"] = price
                                self.tickers[symbol]["bid"] = price * 0.999
                                self.tickers[symbol]["ask"] = price * 1.001
                                self.tickers[symbol]["change"] = price - old_price
                                self.tickers[symbol]["percentage"] = ((price / old_price) - 1) * 100
                                self.tickers[symbol]["timestamp"] = int(time.time() * 1000)
                                self.tickers[symbol]["datetime"] = datetime.now().isoformat()
                                
                                # Actualizar máximo y mínimo
                                if price > self.tickers[symbol]["high"]:
                                    self.tickers[symbol]["high"] = price
                                    
                                if price < self.tickers[symbol]["low"]:
                                    self.tickers[symbol]["low"] = price
                                    
                                # Incrementar volumen
                                volume_increment = self.config["default_volume"] * 0.001 * random.uniform(0.5, 1.5)
                                self.tickers[symbol]["volume"] += volume_increment
                                self.tickers[symbol]["quoteVolume"] += volume_increment * price
                                
                            # Actualizar orderbook ocasionalmente (cada ~10 ticks)
                            if random.random() < 0.1:
                                self.orderbooks[symbol] = self._generate_orderbook(
                                    symbol, price, self.config["liquidity_depth"]
                                )
                                
                            # Notificar a suscriptores
                            await self._notify_ticker_update(symbol)
                            
                    except Exception as e:
                        self.logger.error(f"Error actualizando {symbol}: {e}")
                        self.error_count += 1
                
                # Esperar hasta el siguiente tick
                await asyncio.sleep(self.config["tick_interval_ms"] / 1000)
                
            except asyncio.CancelledError:
                self.logger.info("Tarea de actualización de tickers cancelada")
                break
                
            except Exception as e:
                self.logger.error(f"Error en tarea de actualización: {e}")
                await asyncio.sleep(1)  # Esperar un segundo antes de reintentar
                
        self.logger.info("Tarea de actualización de tickers detenida")
    
    async def _event_processor(self) -> None:
        """Tarea de fondo para procesar eventos programados."""
        self.logger.info("Iniciando procesador de eventos")
        
        while self.running:
            try:
                # Eventos pendientes de ejecución
                current_time = time.time()
                
                for event in self.events:
                    # Procesar solo eventos no ejecutados cuyo momento ha llegado
                    if not event["executed"] and event["time"] <= current_time:
                        await self._process_event(event)
                        event["executed"] = True
                        self.event_count += 1
                
                # Eliminar eventos ya ejecutados de la lista
                self.events = [e for e in self.events if not e["executed"]]
                
                # Esperar un momento antes de la próxima comprobación
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                self.logger.info("Procesador de eventos cancelado")
                break
                
            except Exception as e:
                self.logger.error(f"Error en procesador de eventos: {e}")
                await asyncio.sleep(1)  # Esperar un segundo antes de reintentar
                
        self.logger.info("Procesador de eventos detenido")
    
    async def _process_event(self, event: Dict[str, Any]) -> None:
        """
        Procesar un evento de mercado.
        
        Args:
            event: Datos del evento
        """
        symbol = event["symbol"]
        event_type = event["type"]
        impact = event["impact"]
        
        self.logger.info(f"Procesando evento {event_type.name} para {symbol} con impacto {impact:.2%}")
        
        if symbol not in self.tickers:
            self.logger.warning(f"Símbolo {symbol} no encontrado para evento")
            return
            
        # Obtener precio actual
        current_price = self.tickers[symbol]["price"]
        
        # Calcular nuevo precio según tipo de evento
        if event_type == MarketEventType.POSITIVE_NEWS:
            # Noticias positivas: subida de precio
            new_price = current_price * (1 + impact)
            
        elif event_type == MarketEventType.NEGATIVE_NEWS:
            # Noticias negativas: bajada de precio
            new_price = current_price * (1 - impact)
            
        elif event_type == MarketEventType.WHALE_BUY:
            # Compra de ballena: subida rápida de precio
            new_price = current_price * (1 + impact * 1.5)
            
        elif event_type == MarketEventType.WHALE_SELL:
            # Venta de ballena: bajada rápida de precio
            new_price = current_price * (1 - impact * 1.5)
            
        elif event_type == MarketEventType.LIQUIDITY_SPIKE:
            # Aumento de liquidez: reducción de volatilidad
            new_price = current_price
            if symbol in self.market_drivers:
                self.market_drivers[symbol]["volatility"] *= 0.5
                
        elif event_type == MarketEventType.LIQUIDITY_DRAIN:
            # Disminución de liquidez: aumento de volatilidad
            new_price = current_price
            if symbol in self.market_drivers:
                self.market_drivers[symbol]["volatility"] *= 2.0
                
        elif event_type == MarketEventType.VOLATILITY_INCREASE:
            # Aumento de volatilidad
            new_price = current_price
            if symbol in self.market_drivers:
                self.market_drivers[symbol]["volatility"] *= 3.0
                
        elif event_type == MarketEventType.VOLATILITY_DECREASE:
            # Disminución de volatilidad
            new_price = current_price
            if symbol in self.market_drivers:
                self.market_drivers[symbol]["volatility"] *= 0.3
                
        elif event_type == MarketEventType.TECHNICAL_BREAKDOWN:
            # Problemas técnicos: generar errores en próximas operaciones
            new_price = current_price
            # Aumentar temporalmente la tasa de errores del simulador
            self.config["error_rate"] = 0.5
            
            # Programar un evento para restaurar la tasa de errores
            async def restore_error_rate():
                await asyncio.sleep(60)  # Después de 1 minuto
                self.config["error_rate"] = self.default_config["error_rate"]
                
            task = asyncio.create_task(restore_error_rate())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            
        else:
            # Evento desconocido: cambio de precio aleatorio
            direction = random.choice([-1, 1])
            new_price = current_price * (1 + direction * impact)
            
        # Si existe el driver de mercado, registrar evento y su impacto
        if symbol in self.market_drivers:
            driver = self.market_drivers[symbol]
            driver["active_events"].append({
                "type": event_type,
                "impact": impact,
                "start_time": time.time(),
                "duration": random.uniform(10, 300)  # 10 segundos a 5 minutos
            })
            
        # Actualizar precio inmediatamente
        await self.set_ticker_price(symbol, new_price)
    
    async def _update_market_price(self, symbol: str, driver: Dict[str, Any]) -> float:
        """
        Actualizar precio de mercado según patrón y emociones.
        
        Args:
            symbol: Símbolo
            driver: Configuración del driver
            
        Returns:
            Nuevo precio
        """
        # Obtener tiempo actual
        current_time = time.time()
        time_delta = current_time - driver["last_update"]
        driver["last_update"] = current_time
        
        # Comprobar si es momento de cambiar el patrón
        pattern_elapsed = current_time - driver["pattern_start_time"]
        if pattern_elapsed > driver["pattern_duration"]:
            # Elegir nuevo patrón (con cierta probabilidad de continuar)
            if random.random() < 0.7:  # 70% de probabilidad de cambiar
                new_pattern = random.choice([p for p in MarketPattern if p != driver["pattern"]])
                driver["pattern"] = new_pattern
                self.pattern_changes += 1
                self.logger.debug(f"Patrón de {symbol} cambiado a {new_pattern.name}")
                
            # Reiniciar tiempo de patrón
            driver["pattern_start_time"] = current_time
            driver["pattern_duration"] = random.uniform(
                0.5 * self.config["pattern_duration"],
                1.5 * self.config["pattern_duration"]
            )
            
        # Obtener precio actual
        current_price = driver["current_price"]
        
        # Calcular base de volatilidad (escalar por tiempo desde última actualización)
        base_volatility = driver["volatility"] * (time_delta / 1.0)
        
        # Ajustar volatilidad según patrón
        if driver["pattern"] == MarketPattern.VOLATILE:
            volatility = base_volatility * 3
        elif driver["pattern"] == MarketPattern.CONSOLIDATION:
            volatility = base_volatility * 0.2
        else:
            volatility = base_volatility
            
        # Generar cambio base según patrón
        if driver["pattern"] == MarketPattern.TRENDING_UP:
            base_change = volatility * driver["trend_strength"] * random.uniform(0.5, 1.5)
        elif driver["pattern"] == MarketPattern.TRENDING_DOWN:
            base_change = -volatility * driver["trend_strength"] * random.uniform(0.5, 1.5)
        elif driver["pattern"] == MarketPattern.RANDOM_WALK:
            base_change = volatility * random.uniform(-1, 1)
        elif driver["pattern"] == MarketPattern.FLASH_CRASH and random.random() < 0.05:
            base_change = -volatility * 5
        elif driver["pattern"] == MarketPattern.FLASH_PUMP and random.random() < 0.05:
            base_change = volatility * 5
        elif driver["pattern"] == MarketPattern.BREAKOUT:
            # Romper un nivel de soporte o resistencia
            closest_resistance = min((level for level in driver["resistance_levels"] if level > current_price), 
                                    default=current_price * 1.05)
            closest_support = max((level for level in driver["support_levels"] if level < current_price),
                                default=current_price * 0.95)
            
            if random.random() < 0.6:  # Mayor probabilidad de ruptura alcista
                # Ruptura alcista
                base_change = (closest_resistance - current_price) * random.uniform(0.1, 0.3)
            else:
                # Ruptura bajista
                base_change = (closest_support - current_price) * random.uniform(0.1, 0.3)
        elif driver["pattern"] == MarketPattern.LIQUIDITY_ZONE:
            # Oscilar alrededor de una zona de liquidez
            liquid_zone = driver["base_price"] * random.uniform(0.97, 1.03)
            direction = 1 if liquid_zone > current_price else -1
            base_change = direction * volatility * 0.5
        else:
            base_change = volatility * random.uniform(-1, 1)
            
        # Factores emocionales
        emotion_impact = 0.0
        
        # Influencia del miedo (movimientos bruscos a la baja)
        fear_factor = driver["emotion_factors"][HumanEmotionFactor.FEAR]
        if random.random() < fear_factor * 0.1:
            emotion_impact -= volatility * random.uniform(1, 3) * fear_factor
            
        # Influencia de la codicia (acumulación)
        greed_factor = driver["emotion_factors"][HumanEmotionFactor.GREED]
        if random.random() < greed_factor * 0.1:
            emotion_impact += volatility * random.uniform(0.5, 1.5) * greed_factor
            
        # Influencia del FOMO (compras frenéticas)
        fomo_factor = driver["emotion_factors"][HumanEmotionFactor.FOMO]
        if driver["pattern"] == MarketPattern.TRENDING_UP and random.random() < fomo_factor * 0.2:
            emotion_impact += volatility * random.uniform(1, 4) * fomo_factor
            
        # Influencia del pánico (ventas masivas)
        panic_factor = driver["emotion_factors"][HumanEmotionFactor.PANIC]
        if driver["pattern"] == MarketPattern.TRENDING_DOWN and random.random() < panic_factor * 0.2:
            emotion_impact -= volatility * random.uniform(2, 5) * panic_factor
            
        # Calcular impacto de eventos activos
        event_impact = 0.0
        events_to_remove = []
        
        for event_data in driver["active_events"]:
            # Verificar si el evento sigue activo
            event_age = current_time - event_data["start_time"]
            if event_age > event_data["duration"]:
                events_to_remove.append(event_data)
                continue
                
            # Calcular impacto decreciente con el tiempo
            decay_factor = event_data["impact"] * math.exp(-event_age / event_data["duration"])
            
            # Impacto según tipo de evento
            if event_data["type"] in [MarketEventType.POSITIVE_NEWS, MarketEventType.WHALE_BUY]:
                event_impact += decay_factor * current_price
            elif event_data["type"] in [MarketEventType.NEGATIVE_NEWS, MarketEventType.WHALE_SELL]:
                event_impact -= decay_factor * current_price
                
        # Eliminar eventos caducados
        for event_data in events_to_remove:
            driver["active_events"].remove(event_data)
            
        # Combinar todos los factores
        total_change = base_change + emotion_impact + event_impact
        
        # Aplicar momentum (inercia del mercado)
        driver["momentum"] = driver["momentum"] * 0.95 + total_change * 0.05
        total_change = total_change * 0.8 + driver["momentum"] * 0.2
        
        # Calcular nuevo precio
        new_price = current_price * (1 + total_change)
        
        # Actualizar precio en el driver
        driver["current_price"] = new_price
        
        return new_price
    
    def _add_order_to_orderbook(self, symbol: str, order: Dict[str, Any]) -> None:
        """
        Añadir una orden al libro de órdenes.
        
        Args:
            symbol: Símbolo
            order: Datos de la orden
        """
        if symbol not in self.orderbooks:
            return
            
        price = order["price"]
        quantity = order["remaining"]
        
        if order["side"].lower() == "buy":
            # Añadir a bids
            self.orderbooks[symbol]["bids"].append([price, quantity])
            # Ordenar bids de mayor a menor precio
            self.orderbooks[symbol]["bids"].sort(reverse=True)
        else:
            # Añadir a asks
            self.orderbooks[symbol]["asks"].append([price, quantity])
            # Ordenar asks de menor a mayor precio
            self.orderbooks[symbol]["asks"].sort()
            
        # Actualizar timestamp
        self.orderbooks[symbol]["timestamp"] = int(time.time() * 1000)
        self.orderbooks[symbol]["datetime"] = datetime.now().isoformat()
        self.orderbooks[symbol]["nonce"] = self.tick_count
    
    def _remove_order_from_orderbook(self, symbol: str, order: Dict[str, Any]) -> None:
        """
        Eliminar una orden del libro de órdenes.
        
        Args:
            symbol: Símbolo
            order: Datos de la orden
        """
        if symbol not in self.orderbooks:
            return
            
        price = order["price"]
        
        if order["side"].lower() == "buy":
            # Buscar y eliminar de bids
            for i, bid in enumerate(self.orderbooks[symbol]["bids"]):
                if abs(bid[0] - price) < 0.000001:
                    self.orderbooks[symbol]["bids"].pop(i)
                    break
        else:
            # Buscar y eliminar de asks
            for i, ask in enumerate(self.orderbooks[symbol]["asks"]):
                if abs(ask[0] - price) < 0.000001:
                    self.orderbooks[symbol]["asks"].pop(i)
                    break
                    
        # Actualizar timestamp
        self.orderbooks[symbol]["timestamp"] = int(time.time() * 1000)
        self.orderbooks[symbol]["datetime"] = datetime.now().isoformat()
        self.orderbooks[symbol]["nonce"] = self.tick_count
    
    async def _notify_ticker_update(self, symbol: str) -> None:
        """
        Notificar actualización de ticker a suscriptores.
        
        Args:
            symbol: Símbolo actualizado
        """
        if not self.websocket_running:
            return
            
        # Buscar canales relevantes
        symbol_channel = f"ticker:{symbol}"
        all_tickers_channel = "ticker:all"
        
        channels_to_notify = []
        if symbol_channel in self.listeners:
            channels_to_notify.append(symbol_channel)
            
        if all_tickers_channel in self.listeners:
            channels_to_notify.append(all_tickers_channel)
            
        # Sin suscriptores, salir
        if not channels_to_notify:
            return
            
        # Datos para notificación
        ticker_data = self.tickers[symbol].copy()
        ticker_data["update_id"] = self.tick_count
        ticker_data["update_time"] = int(time.time() * 1000)
        
        # Notificar a todos los suscriptores
        for channel in channels_to_notify:
            for callback in self.listeners.get(channel, []):
                try:
                    # Ejecutar callback de forma segura
                    task = asyncio.create_task(callback(ticker_data))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                except Exception as e:
                    self.logger.error(f"Error notificando a suscriptor: {e}")
                    
    async def _simulate_latency(self) -> None:
        """Simular latencia de red."""
        base_latency = self.config["base_latency_ms"] / 1000  # Convertir a segundos
        variance = self.config["latency_variance"]
        
        # Calcular latencia con variación
        latency = base_latency * (1 + random.uniform(-variance, variance))
        
        await asyncio.sleep(latency)


# Clase de fábrica para crear simuladores de exchanges
class ExchangeSimulatorFactory:
    """Fábrica para crear instancias de simuladores de exchanges."""
    
    @staticmethod
    async def create_simulator(exchange_id: str, config: Optional[Dict[str, Any]] = None) -> ExchangeSimulator:
        """
        Crear y configurar un simulador de exchange.
        
        Args:
            exchange_id: Identificador del exchange
            config: Configuración opcional
            
        Returns:
            Instancia de ExchangeSimulator configurada
        """
        simulator = ExchangeSimulator(exchange_id, config)
        await simulator.initialize()
        return simulator