"""
Módulo de Ejecución para el Pipeline de Genesis.

Este módulo se encarga de ejecutar las órdenes de trading generadas por el módulo
de decisión, incluyendo comunicación con exchanges y seguimiento de órdenes.
"""
import logging
import time
import json
import hmac
import hashlib
import base64
import uuid
import random
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta

import aiohttp

from genesis.base import GenesisComponent, GenesisSingleton, validate_mode
from genesis.db.transcendental_database import TranscendentalDatabase

# Configuración de logging
logger = logging.getLogger("genesis.pipeline.execution")

class Order:
    """Orden de trading con información completa."""
    
    def __init__(self, 
                symbol: str, 
                side: str, 
                order_type: str,
                quantity: float,
                price: Optional[float] = None):
        """
        Inicializar orden de trading.
        
        Args:
            symbol: Símbolo de trading
            side: Lado (buy/sell)
            order_type: Tipo de orden (market/limit/stop/etc)
            quantity: Cantidad a operar
            price: Precio para órdenes limit/stop (opcional)
        """
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.timestamp = time.time()
        self.id = f"order_{int(self.timestamp)}_{random.randint(1000, 9999)}"
        self.status = "created"
        self.filled_quantity = 0.0
        self.average_price = None
        self.exchange_order_id = None
        self.fees = 0.0
        self.fees_currency = None
        self.execution_time = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir orden a diccionario.
        
        Returns:
            Representación como diccionario
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "exchange_order_id": self.exchange_order_id,
            "fees": self.fees,
            "fees_currency": self.fees_currency,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "error": self.error,
            "datetime": datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """
        Crear orden desde diccionario.
        
        Args:
            data: Diccionario con datos de orden
            
        Returns:
            Instancia de Order
        """
        order = cls(
            symbol=data["symbol"],
            side=data["side"],
            order_type=data["order_type"],
            quantity=data["quantity"],
            price=data.get("price")
        )
        order.id = data.get("id", order.id)
        order.status = data.get("status", order.status)
        order.filled_quantity = data.get("filled_quantity", order.filled_quantity)
        order.average_price = data.get("average_price", order.average_price)
        order.exchange_order_id = data.get("exchange_order_id", order.exchange_order_id)
        order.fees = data.get("fees", order.fees)
        order.fees_currency = data.get("fees_currency", order.fees_currency)
        order.timestamp = data.get("timestamp", order.timestamp)
        order.execution_time = data.get("execution_time", order.execution_time)
        order.error = data.get("error", order.error)
        
        return order

class Exchange(GenesisComponent):
    """Interfaz base para exchanges con capacidades trascendentales."""
    
    def __init__(self, exchange_id: str, mode: str = "SINGULARITY_V4"):
        """
        Inicializar interfaz de exchange.
        
        Args:
            exchange_id: Identificador del exchange
            mode: Modo trascendental
        """
        super().__init__(f"exchange_{exchange_id}", mode)
        self.exchange_id = exchange_id
        self.api_key = None
        self.api_secret = None
        self.last_order_time = 0
        self.order_count = 0
        self.db = TranscendentalDatabase()
        
        # Registro para rate limiting
        self.request_timestamps: List[float] = []
        self.request_weights: List[int] = []
        self.rate_limit_window = 60  # Ventana en segundos
        self.max_requests_per_window = 1200  # Máximo de solicitudes por ventana
        
        logger.info(f"Interfaz para exchange {exchange_id} inicializada")
    
    def set_credentials(self, api_key: str, api_secret: str) -> None:
        """
        Establecer credenciales de API.
        
        Args:
            api_key: Clave de API
            api_secret: Secreto de API
        """
        self.api_key = api_key
        self.api_secret = api_secret
        logger.info(f"Credenciales establecidas para {self.exchange_id}")
    
    async def _check_rate_limit(self, weight: int = 1) -> bool:
        """
        Comprobar y gestionar rate limiting.
        
        Args:
            weight: Peso de la solicitud (algunas APIs usan pesos)
            
        Returns:
            True si se puede realizar la solicitud
        """
        now = time.time()
        
        # Eliminar registros antiguos fuera de la ventana
        cutoff = now - self.rate_limit_window
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff]
        self.request_weights = self.request_weights[-len(self.request_timestamps):]
        
        # Calcular peso total en la ventana actual
        total_weight = sum(self.request_weights)
        
        # Comprobar si excedemos el límite
        if total_weight + weight > self.max_requests_per_window:
            wait_time = self.rate_limit_window - (now - self.request_timestamps[0])
            
            if wait_time > 0:
                logger.warning(f"Rate limit alcanzado, esperando {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            # Recomprobar después de esperar
            return await self._check_rate_limit(weight)
        
        # Registrar nueva solicitud
        self.request_timestamps.append(now)
        self.request_weights.append(weight)
        
        return True
    
    async def create_order(self, order: Order) -> Order:
        """
        Crear orden en el exchange.
        
        Args:
            order: Orden a crear
            
        Returns:
            Orden con información de ejecución
        """
        raise NotImplementedError("Las subclases deben implementar create_order")
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Obtener estado de una orden.
        
        Args:
            order_id: ID de la orden en el exchange
            symbol: Símbolo de trading
            
        Returns:
            Estado de la orden
        """
        raise NotImplementedError("Las subclases deben implementar get_order_status")
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancelar orden existente.
        
        Args:
            order_id: ID de la orden en el exchange
            symbol: Símbolo de trading
            
        Returns:
            True si se canceló correctamente
        """
        raise NotImplementedError("Las subclases deben implementar cancel_order")
    
    async def get_account_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener balance de la cuenta.
        
        Returns:
            Balance por moneda
        """
        raise NotImplementedError("Las subclases deben implementar get_account_balance")

class BinanceExchange(Exchange):
    """Implementación de Exchange para Binance."""
    
    def __init__(self, testnet: bool = False, mode: str = "SINGULARITY_V4"):
        """
        Inicializar interfaz para Binance.
        
        Args:
            testnet: Usar testnet en lugar de producción
            mode: Modo trascendental
        """
        super().__init__("binance", mode)
        self.testnet = testnet
        
        # URLs base
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        self.api_version = "v3"
        
        # Endpoints
        self.endpoints = {
            "order": "/api/v3/order",
            "account": "/api/v3/account",
            "exchange_info": "/api/v3/exchangeInfo"
        }
        
        # Parámetros específicos
        self.recv_window = 5000  # ms
        self.max_requests_per_window = 1200  # Límite de Binance
        
        logger.info(f"Interfaz para Binance {'Testnet' if testnet else 'Producción'} inicializada")
    
    def _generate_signature(self, query_string: str) -> str:
        """
        Generar firma para solicitud autenticada.
        
        Args:
            query_string: Cadena de consulta completa
            
        Returns:
            Firma HMAC-SHA256
        """
        if not self.api_secret:
            raise ValueError("API Secret no configurado")
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def create_order(self, order: Order) -> Order:
        """
        Crear orden en Binance.
        
        Args:
            order: Orden a crear
            
        Returns:
            Orden con información de ejecución
        """
        if not self.api_key or not self.api_secret:
            order.status = "failed"
            order.error = "API credentials not set"
            return order
        
        # Verificar rate limit
        await self._check_rate_limit(weight=1)
        
        start_time = time.time()
        
        try:
            # Preparar parámetros
            params = {
                "symbol": order.symbol,
                "side": order.side.upper(),
                "type": order.order_type.upper(),
                "quantity": f"{order.quantity:.8f}".rstrip('0').rstrip('.'),
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            # Añadir precio para órdenes limit
            if order.order_type.lower() == "limit" and order.price:
                params["price"] = f"{order.price:.8f}".rstrip('0').rstrip('.')
                params["timeInForce"] = "GTC"  # Good Till Cancelled
            
            # Construir query string
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            
            # Generar firma
            signature = self._generate_signature(query_string)
            query_string = f"{query_string}&signature={signature}"
            
            # URL completa
            url = f"{self.base_url}{self.endpoints['order']}"
            
            # Cabeceras
            headers = {
                "X-MBX-APIKEY": self.api_key
            }
            
            # Hacer solicitud
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Actualizar orden con información recibida
                        order.status = "active" if result.get("status") == "NEW" else result.get("status", "unknown").lower()
                        order.exchange_order_id = result.get("orderId")
                        order.filled_quantity = float(result.get("executedQty", 0))
                        
                        if float(result.get("executedQty", 0)) > 0:
                            order.average_price = float(result.get("price", 0))
                        
                        order.execution_time = time.time() - start_time
                        logger.info(f"Orden creada en Binance: {order.id} ({order.symbol}, {order.side})")
                    else:
                        error_text = await response.text()
                        order.status = "failed"
                        order.error = f"HTTP {response.status}: {error_text}"
                        logger.error(f"Error al crear orden en Binance: {order.error}")
            
            self.order_count += 1
            self.last_order_time = time.time()
            
            # Registrar operación
            self.register_operation(response.status == 200)
            
            return order
        
        except Exception as e:
            order.status = "failed"
            order.error = str(e)
            order.execution_time = time.time() - start_time
            logger.error(f"Error al crear orden en Binance: {str(e)}")
            
            # Registrar operación fallida
            self.register_operation(False)
            
            return order
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Obtener estado de una orden en Binance.
        
        Args:
            order_id: ID de la orden en Binance
            symbol: Símbolo de trading
            
        Returns:
            Estado de la orden
        """
        if not self.api_key or not self.api_secret:
            return {"status": "error", "message": "API credentials not set"}
        
        # Verificar rate limit
        await self._check_rate_limit(weight=1)
        
        try:
            # Preparar parámetros
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            # Construir query string
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            
            # Generar firma
            signature = self._generate_signature(query_string)
            query_string = f"{query_string}&signature={signature}"
            
            # URL completa
            url = f"{self.base_url}{self.endpoints['order']}"
            
            # Cabeceras
            headers = {
                "X-MBX-APIKEY": self.api_key
            }
            
            # Hacer solicitud
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Convertir a formato estándar
                        status_map = {
                            "NEW": "active",
                            "PARTIALLY_FILLED": "partially_filled",
                            "FILLED": "filled",
                            "CANCELED": "cancelled",
                            "REJECTED": "rejected",
                            "EXPIRED": "expired"
                        }
                        
                        return {
                            "symbol": result.get("symbol"),
                            "order_id": result.get("orderId"),
                            "status": status_map.get(result.get("status"), "unknown"),
                            "side": result.get("side").lower(),
                            "order_type": result.get("type").lower(),
                            "price": float(result.get("price", 0)),
                            "quantity": float(result.get("origQty", 0)),
                            "filled_quantity": float(result.get("executedQty", 0)),
                            "average_price": float(result.get("price", 0)),
                            "timestamp": result.get("time") / 1000 if "time" in result else 0
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Error al obtener estado de orden en Binance: HTTP {response.status}: {error_text}")
                        return {"status": "error", "message": f"HTTP {response.status}: {error_text}"}
            
        except Exception as e:
            logger.error(f"Error al obtener estado de orden en Binance: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancelar orden en Binance.
        
        Args:
            order_id: ID de la orden en Binance
            symbol: Símbolo de trading
            
        Returns:
            True si se canceló correctamente
        """
        if not self.api_key or not self.api_secret:
            return False
        
        # Verificar rate limit
        await self._check_rate_limit(weight=1)
        
        try:
            # Preparar parámetros
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            # Construir query string
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            
            # Generar firma
            signature = self._generate_signature(query_string)
            query_string = f"{query_string}&signature={signature}"
            
            # URL completa
            url = f"{self.base_url}{self.endpoints['order']}"
            
            # Cabeceras
            headers = {
                "X-MBX-APIKEY": self.api_key
            }
            
            # Hacer solicitud
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Orden cancelada en Binance: {order_id} ({symbol})")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Error al cancelar orden en Binance: HTTP {response.status}: {error_text}")
                        return False
            
        except Exception as e:
            logger.error(f"Error al cancelar orden en Binance: {str(e)}")
            return False
    
    async def get_account_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener balance de la cuenta en Binance.
        
        Returns:
            Balance por moneda
        """
        if not self.api_key or not self.api_secret:
            return {}
        
        # Verificar rate limit
        await self._check_rate_limit(weight=5)
        
        try:
            # Preparar parámetros
            params = {
                "timestamp": int(time.time() * 1000),
                "recvWindow": self.recv_window
            }
            
            # Construir query string
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            
            # Generar firma
            signature = self._generate_signature(query_string)
            query_string = f"{query_string}&signature={signature}"
            
            # URL completa
            url = f"{self.base_url}{self.endpoints['account']}"
            
            # Cabeceras
            headers = {
                "X-MBX-APIKEY": self.api_key
            }
            
            # Hacer solicitud
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Procesar balances
                        balances = {}
                        for asset in result.get("balances", []):
                            asset_name = asset.get("asset")
                            free = float(asset.get("free", 0))
                            locked = float(asset.get("locked", 0))
                            
                            if free > 0 or locked > 0:
                                balances[asset_name] = {
                                    "free": free,
                                    "locked": locked,
                                    "total": free + locked
                                }
                        
                        return balances
                    else:
                        error_text = await response.text()
                        logger.error(f"Error al obtener balance en Binance: HTTP {response.status}: {error_text}")
                        return {}
            
        except Exception as e:
            logger.error(f"Error al obtener balance en Binance: {str(e)}")
            return {}

class OrderManager(GenesisComponent):
    """
    Gestor de órdenes con capacidades trascendentales.
    
    Este componente gestiona la creación, seguimiento y actualización de órdenes,
    así como la reconciliación con el estado del exchange.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar gestor de órdenes.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("order_manager", mode)
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Dict[str, Any]] = []
        self.db = TranscendentalDatabase()
        self.exchanges: Dict[str, Exchange] = {}
        self.default_exchange = None
        
        logger.info(f"Gestor de órdenes inicializado en modo {mode}")
    
    def register_exchange(self, exchange: Exchange, default: bool = False) -> None:
        """
        Registrar exchange para operaciones.
        
        Args:
            exchange: Instancia del exchange
            default: Establecer como exchange por defecto
        """
        self.exchanges[exchange.exchange_id] = exchange
        
        if default or self.default_exchange is None:
            self.default_exchange = exchange.exchange_id
            
        logger.info(f"Exchange {exchange.exchange_id} registrado. Default: {default}")
    
    async def create_order(self, 
                         symbol: str, 
                         side: str, 
                         order_type: str,
                         quantity: float,
                         price: Optional[float] = None,
                         exchange_id: Optional[str] = None) -> Order:
        """
        Crear y enviar orden a un exchange.
        
        Args:
            symbol: Símbolo de trading
            side: Lado (buy/sell)
            order_type: Tipo de orden (market/limit/stop/etc)
            quantity: Cantidad a operar
            price: Precio para órdenes limit/stop (opcional)
            exchange_id: ID del exchange a utilizar (opcional)
            
        Returns:
            Orden creada con información de ejecución
        """
        # Determinar exchange a utilizar
        exchange_id = exchange_id or self.default_exchange
        
        if not exchange_id or exchange_id not in self.exchanges:
            raise ValueError(f"Exchange no válido: {exchange_id}")
        
        exchange = self.exchanges[exchange_id]
        
        # Crear orden
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
        
        # Enviar orden al exchange
        order = await exchange.create_order(order)
        
        # Registrar en el gestor
        self.orders[order.id] = order
        self.order_history.append(order.to_dict())
        
        # Guardar en base de datos
        await self._save_order(order)
        
        logger.info(f"Orden {order.id} creada: {symbol} {side} {quantity} a {price if price else 'market'}")
        return order
    
    async def _save_order(self, order: Order) -> bool:
        """
        Guardar orden en base de datos.
        
        Args:
            order: Orden a guardar
            
        Returns:
            True si se guardó correctamente
        """
        try:
            # Guardar en base de datos trascendental
            await self.db.store("orders", order.id, order.to_dict())
            return True
        except Exception as e:
            logger.error(f"Error al guardar orden en DB: {str(e)}")
            return False
    
    async def update_order_status(self, order_id: str) -> Optional[Order]:
        """
        Actualizar estado de una orden desde el exchange.
        
        Args:
            order_id: ID de la orden en el gestor
            
        Returns:
            Orden actualizada o None si no existe
        """
        if order_id not in self.orders:
            logger.warning(f"Orden no encontrada: {order_id}")
            return None
        
        order = self.orders[order_id]
        
        # Verificar si tenemos exchange_order_id
        if not order.exchange_order_id:
            logger.warning(f"Orden {order_id} no tiene exchange_order_id")
            return order
        
        # Determinar exchange a utilizar
        exchange_id = self.default_exchange
        
        if not exchange_id or exchange_id not in self.exchanges:
            logger.error(f"Exchange no válido: {exchange_id}")
            return order
        
        exchange = self.exchanges[exchange_id]
        
        # Obtener estado actualizado
        order_status = await exchange.get_order_status(order.exchange_order_id, order.symbol)
        
        if "status" in order_status and order_status["status"] != "error":
            # Actualizar orden
            order.status = order_status["status"]
            order.filled_quantity = order_status.get("filled_quantity", order.filled_quantity)
            
            if "average_price" in order_status and order_status["average_price"] > 0:
                order.average_price = order_status["average_price"]
            
            # Guardar actualización en base de datos
            await self._save_order(order)
            
            logger.info(f"Orden {order_id} actualizada: {order.status}, llenado: {order.filled_quantity}/{order.quantity}")
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancelar una orden existente.
        
        Args:
            order_id: ID de la orden en el gestor
            
        Returns:
            True si se canceló correctamente
        """
        if order_id not in self.orders:
            logger.warning(f"Orden no encontrada para cancelar: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        # Verificar si se puede cancelar
        if order.status not in ["active", "partially_filled"]:
            logger.warning(f"Orden {order_id} no se puede cancelar, estado: {order.status}")
            return False
        
        # Verificar si tenemos exchange_order_id
        if not order.exchange_order_id:
            logger.warning(f"Orden {order_id} no tiene exchange_order_id")
            return False
        
        # Determinar exchange a utilizar
        exchange_id = self.default_exchange
        
        if not exchange_id or exchange_id not in self.exchanges:
            logger.error(f"Exchange no válido: {exchange_id}")
            return False
        
        exchange = self.exchanges[exchange_id]
        
        # Cancelar orden
        success = await exchange.cancel_order(order.exchange_order_id, order.symbol)
        
        if success:
            # Actualizar estado
            order.status = "cancelled"
            
            # Guardar actualización en base de datos
            await self._save_order(order)
            
            logger.info(f"Orden {order_id} cancelada")
        
        return success
    
    async def get_active_orders(self) -> List[Order]:
        """
        Obtener órdenes activas.
        
        Returns:
            Lista de órdenes activas
        """
        active_orders = [
            order for order in self.orders.values() 
            if order.status in ["active", "partially_filled"]
        ]
        
        return active_orders
    
    async def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de órdenes.
        
        Args:
            limit: Número máximo de órdenes a retornar
            
        Returns:
            Lista de órdenes históricas
        """
        history = self.order_history[-limit:] if limit > 0 else self.order_history
        
        # Ordenar por fecha (más recientes primero)
        history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return history
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del gestor de órdenes.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Calcular estadísticas adicionales
        total_orders = len(self.order_history)
        active_orders = sum(1 for o in self.orders.values() if o.status in ["active", "partially_filled"])
        filled_orders = sum(1 for o in self.orders.values() if o.status == "filled")
        
        # Calcular volumen operado
        total_volume = sum(
            (o.filled_quantity * (o.average_price or 0)) 
            for o in self.orders.values() 
            if o.status in ["filled", "partially_filled"] and o.average_price
        )
        
        order_stats = {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled_orders,
            "total_volume": total_volume,
            "exchanges": list(self.exchanges.keys()),
            "default_exchange": self.default_exchange
        }
        
        stats.update(order_stats)
        return stats

class SimulatedExchange(Exchange):
    """Implementación de Exchange simulado para pruebas y desarrollo."""
    
    def __init__(self, exchange_id: str = "simulated", mode: str = "SINGULARITY_V4"):
        """
        Inicializar exchange simulado.
        
        Args:
            exchange_id: Identificador del exchange
            mode: Modo trascendental
        """
        super().__init__(exchange_id, mode)
        
        # Simular balance inicial
        self.balances = {
            "USDT": {"free": 10000.0, "locked": 0.0, "total": 10000.0},
            "BTC": {"free": 0.1, "locked": 0.0, "total": 0.1},
            "ETH": {"free": 1.0, "locked": 0.0, "total": 1.0}
        }
        
        # Órdenes simuladas
        self.simulated_orders: Dict[str, Dict[str, Any]] = {}
        
        # Precios simulados
        self.simulated_prices = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "BNBUSDT": 400.0
        }
        
        # Volatilidad simulada (% de variación máxima)
        self.volatility = 0.01  # 1%
        
        # Tarifas simuladas
        self.fees = 0.001  # 0.1%
        
        logger.info(f"Exchange simulado {exchange_id} inicializado")
    
    def _update_simulated_prices(self) -> None:
        """Actualizar precios simulados con movimiento aleatorio."""
        for symbol, price in self.simulated_prices.items():
            # Variación aleatoria entre -volatilidad y +volatilidad
            variation = (random.random() * 2 - 1) * self.volatility
            new_price = price * (1 + variation)
            self.simulated_prices[symbol] = new_price
    
    async def create_order(self, order: Order) -> Order:
        """
        Crear orden simulada.
        
        Args:
            order: Orden a crear
            
        Returns:
            Orden con información de ejecución simulada
        """
        start_time = time.time()
        
        try:
            # Actualizar precios simulados
            self._update_simulated_prices()
            
            # Verificar si el símbolo existe
            if order.symbol not in self.simulated_prices:
                order.status = "failed"
                order.error = f"Symbol {order.symbol} not found"
                return order
            
            # Obtener precio simulado
            current_price = self.simulated_prices[order.symbol]
            
            # Determinar precio de ejecución
            if order.order_type.lower() == "market":
                # Para órdenes market, usar precio actual con slippage aleatorio
                slippage = (random.random() * 0.002)  # 0-0.2%
                execution_price = current_price * (1 + slippage) if order.side.lower() == "buy" else current_price * (1 - slippage)
            else:
                # Para órdenes limit, usar precio especificado si es alcanzable
                if (order.side.lower() == "buy" and order.price >= current_price) or (order.side.lower() == "sell" and order.price <= current_price):
                    execution_price = order.price
                else:
                    # Orden limit no ejecutable inmediatamente
                    order.status = "active"
                    order.exchange_order_id = str(uuid.uuid4())
                    self.simulated_orders[order.exchange_order_id] = {
                        "order": order.to_dict(),
                        "current_price": current_price
                    }
                    order.execution_time = time.time() - start_time
                    return order
            
            # Calcular fees
            fees = order.quantity * execution_price * self.fees
            
            # Actualizar balances simulados
            if order.side.lower() == "buy":
                base_currency = order.symbol[:-4] if order.symbol.endswith("USDT") else order.symbol.split("/")[0]
                quote_currency = "USDT"
                
                # Verificar balance
                required_amount = order.quantity * execution_price + fees
                if quote_currency in self.balances and self.balances[quote_currency]["free"] >= required_amount:
                    # Actualizar balances
                    self.balances[quote_currency]["free"] -= required_amount
                    
                    if base_currency not in self.balances:
                        self.balances[base_currency] = {"free": 0.0, "locked": 0.0, "total": 0.0}
                    
                    self.balances[base_currency]["free"] += order.quantity
                    self.balances[base_currency]["total"] = self.balances[base_currency]["free"] + self.balances[base_currency]["locked"]
                    self.balances[quote_currency]["total"] = self.balances[quote_currency]["free"] + self.balances[quote_currency]["locked"]
                else:
                    order.status = "failed"
                    order.error = "Insufficient balance"
                    return order
            else:  # sell
                base_currency = order.symbol[:-4] if order.symbol.endswith("USDT") else order.symbol.split("/")[0]
                quote_currency = "USDT"
                
                # Verificar balance
                if base_currency in self.balances and self.balances[base_currency]["free"] >= order.quantity:
                    # Actualizar balances
                    self.balances[base_currency]["free"] -= order.quantity
                    
                    if quote_currency not in self.balances:
                        self.balances[quote_currency] = {"free": 0.0, "locked": 0.0, "total": 0.0}
                    
                    self.balances[quote_currency]["free"] += (order.quantity * execution_price - fees)
                    self.balances[base_currency]["total"] = self.balances[base_currency]["free"] + self.balances[base_currency]["locked"]
                    self.balances[quote_currency]["total"] = self.balances[quote_currency]["free"] + self.balances[quote_currency]["locked"]
                else:
                    order.status = "failed"
                    order.error = "Insufficient balance"
                    return order
            
            # Actualizar orden
            order.status = "filled"
            order.exchange_order_id = str(uuid.uuid4())
            order.filled_quantity = order.quantity
            order.average_price = execution_price
            order.fees = fees
            order.fees_currency = quote_currency
            order.execution_time = time.time() - start_time
            
            # Registrar orden simulada
            self.simulated_orders[order.exchange_order_id] = {
                "order": order.to_dict(),
                "execution_price": execution_price,
                "fees": fees
            }
            
            # Registrar operación
            self.order_count += 1
            self.last_order_time = time.time()
            self.register_operation(True)
            
            logger.info(f"Orden simulada ejecutada: {order.symbol} {order.side} {order.quantity} @ {execution_price:.2f}")
            
            return order
        
        except Exception as e:
            order.status = "failed"
            order.error = str(e)
            order.execution_time = time.time() - start_time
            logger.error(f"Error en orden simulada: {str(e)}")
            
            # Registrar operación fallida
            self.register_operation(False)
            
            return order
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Obtener estado de una orden simulada.
        
        Args:
            order_id: ID de la orden simulada
            symbol: Símbolo de trading
            
        Returns:
            Estado de la orden
        """
        if order_id not in self.simulated_orders:
            return {"status": "error", "message": "Order not found"}
        
        order_data = self.simulated_orders[order_id]
        
        # Si la orden está activa, verificar si se puede ejecutar
        order_dict = order_data["order"]
        
        if order_dict.get("status") == "active":
            # Actualizar precios simulados
            self._update_simulated_prices()
            
            current_price = self.simulated_prices.get(symbol, order_data.get("current_price", 0))
            
            # Verificar si la orden limit se puede ejecutar
            if order_dict.get("order_type", "").lower() == "limit":
                price = order_dict.get("price", 0)
                
                if (order_dict.get("side") == "buy" and current_price <= price) or (order_dict.get("side") == "sell" and current_price >= price):
                    # Ejecutar orden
                    order_dict["status"] = "filled"
                    order_dict["filled_quantity"] = order_dict.get("quantity", 0)
                    order_dict["average_price"] = price
                    
                    logger.info(f"Orden simulada {order_id} ejecutada: {symbol} @ {price}")
        
        return order_dict
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancelar orden simulada.
        
        Args:
            order_id: ID de la orden simulada
            symbol: Símbolo de trading
            
        Returns:
            True si se canceló correctamente
        """
        if order_id not in self.simulated_orders:
            return False
        
        order_data = self.simulated_orders[order_id]
        order_dict = order_data["order"]
        
        if order_dict.get("status") in ["active", "partially_filled"]:
            order_dict["status"] = "cancelled"
            logger.info(f"Orden simulada {order_id} cancelada: {symbol}")
            return True
        else:
            logger.warning(f"No se puede cancelar orden simulada {order_id}, estado: {order_dict.get('status')}")
            return False
    
    async def get_account_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener balance simulado.
        
        Returns:
            Balance por moneda
        """
        # Devolver una copia para evitar modificaciones externas
        return {k: v.copy() for k, v in self.balances.items()}

class ExecutionEngine(GenesisComponent, GenesisSingleton):
    """
    Motor de ejecución con capacidades trascendentales.
    
    Este componente coordina la ejecución de órdenes, la interacción con exchanges
    y el seguimiento de operaciones.
    """
    
    def __init__(self, mode: str = "SINGULARITY_V4"):
        """
        Inicializar motor de ejecución.
        
        Args:
            mode: Modo trascendental
        """
        super().__init__("execution_engine", mode)
        self.order_manager = OrderManager(mode)
        self.db = TranscendentalDatabase()
        
        # Opciones de ejecución
        self.simulation_mode = True
        self.auto_cancel_after = 3600  # Cancelar órdenes después de 1 hora
        
        logger.info(f"Motor de ejecución inicializado en modo {mode}")
    
    def set_simulation_mode(self, enabled: bool) -> None:
        """
        Establecer modo de simulación.
        
        Args:
            enabled: True para habilitar simulación
        """
        self.simulation_mode = enabled
        logger.info(f"Modo de simulación: {'Activado' if enabled else 'Desactivado'}")
    
    async def initialize(self) -> bool:
        """
        Inicializar motor de ejecución.
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            # Inicializar exchanges
            if self.simulation_mode:
                # Usar exchange simulado
                simulated = SimulatedExchange(mode=self.mode)
                self.order_manager.register_exchange(simulated, default=True)
                logger.info("Exchange simulado registrado como predeterminado")
            else:
                # Usar exchange real
                binance = BinanceExchange(testnet=True, mode=self.mode)
                
                # Aquí normalmente estableceríamos las credenciales desde variables de entorno
                # Por ahora, en el pipeline sólo lo inicializamos
                self.order_manager.register_exchange(binance, default=True)
                logger.info("Exchange Binance Testnet registrado como predeterminado")
            
            logger.info("Motor de ejecución inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar motor de ejecución: {str(e)}")
            return False
    
    async def execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar una decisión de trading.
        
        Args:
            decision: Decisión a ejecutar
            
        Returns:
            Resultado de la ejecución
        """
        action = decision.get("action", "unknown")
        symbol = decision.get("symbol", "")
        price = decision.get("price")
        amount = decision.get("amount", 0)
        
        result = {
            "decision_id": decision.get("id", ""),
            "action": action,
            "symbol": symbol,
            "executed": False,
            "timestamp": time.time()
        }
        
        try:
            if action == "buy":
                # Crear orden de compra
                order = await self.order_manager.create_order(
                    symbol=symbol,
                    side="buy",
                    order_type="limit" if price else "market",
                    quantity=amount,
                    price=price
                )
                
                result["executed"] = order.status in ["active", "filled", "partially_filled"]
                result["order_id"] = order.id
                result["exchange_order_id"] = order.exchange_order_id
                result["status"] = order.status
                
                if not result["executed"]:
                    result["error"] = order.error
                
                logger.info(f"Decisión de compra ejecutada: {symbol}, {amount}, resultado: {order.status}")
            
            elif action == "sell" or action == "sell_short":
                # Crear orden de venta
                order = await self.order_manager.create_order(
                    symbol=symbol,
                    side="sell",
                    order_type="limit" if price else "market",
                    quantity=amount,
                    price=price
                )
                
                result["executed"] = order.status in ["active", "filled", "partially_filled"]
                result["order_id"] = order.id
                result["exchange_order_id"] = order.exchange_order_id
                result["status"] = order.status
                
                if not result["executed"]:
                    result["error"] = order.error
                
                logger.info(f"Decisión de venta ejecutada: {symbol}, {amount}, resultado: {order.status}")
            
            elif action == "exit":
                # Para salir, crear orden de venta (si es posición larga) o compra (si es corta)
                position_type = decision.get("position_type", "long")
                side = "sell" if position_type == "long" else "buy"
                
                order = await self.order_manager.create_order(
                    symbol=symbol,
                    side=side,
                    order_type="limit" if price else "market",
                    quantity=amount,
                    price=price
                )
                
                result["executed"] = order.status in ["active", "filled", "partially_filled"]
                result["order_id"] = order.id
                result["exchange_order_id"] = order.exchange_order_id
                result["status"] = order.status
                
                if not result["executed"]:
                    result["error"] = order.error
                
                logger.info(f"Decisión de salida ejecutada: {symbol}, {amount}, resultado: {order.status}")
            
            else:
                result["error"] = f"Acción no soportada: {action}"
                logger.warning(f"Acción de decisión no soportada: {action}")
        
        except Exception as e:
            result["executed"] = False
            result["error"] = str(e)
            logger.error(f"Error al ejecutar decisión: {str(e)}")
        
        return result
    
    async def execute_decisions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar todas las decisiones pendientes.
        
        Args:
            data: Datos con decisiones
            
        Returns:
            Datos con resultados de ejecución
        """
        execution_results = []
        
        # Verificar si hay decisiones pendientes
        new_decisions = []
        
        # Filtrar decisiones que aún no se han ejecutado
        for decision in data.get("decisions", []):
            if not any(r.get("decision_id") == decision.get("id") for r in data.get("execution_results", [])):
                new_decisions.append(decision)
        
        if not new_decisions:
            logger.info("No hay nuevas decisiones para ejecutar")
            return data
        
        logger.info(f"Ejecutando {len(new_decisions)} decisiones pendientes")
        
        # Ejecutar cada decisión
        for decision in new_decisions:
            result = await self.execute_decision(decision)
            execution_results.append(result)
        
        # Actualizar resultados en datos
        if "execution_results" not in data:
            data["execution_results"] = []
        
        data["execution_results"].extend(execution_results)
        data["execution_timestamp"] = time.time()
        
        return data
    
    async def update_order_statuses(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualizar estado de órdenes activas.
        
        Args:
            data: Datos con órdenes
            
        Returns:
            Datos con estados actualizados
        """
        # Obtener órdenes activas
        active_orders = await self.order_manager.get_active_orders()
        
        if not active_orders:
            logger.debug("No hay órdenes activas para actualizar")
            return data
        
        logger.info(f"Actualizando estado de {len(active_orders)} órdenes activas")
        
        # Actualizar cada orden
        for order in active_orders:
            updated_order = await self.order_manager.update_order_status(order.id)
            
            # Verificar si se completó
            if updated_order and updated_order.status == "filled":
                logger.info(f"Orden {order.id} completada: {order.symbol}, {order.filled_quantity}/{order.quantity} @ {order.average_price}")
                
                # Verificar si necesitamos actualizar posiciones
                # (en un sistema real, esto podría desencadenar actualización de posiciones)
        
        # Cancelar órdenes antiguas si es necesario
        await self._cancel_old_orders(active_orders)
        
        # Actualizar lista de órdenes en datos
        orders_dict = {o.id: o.to_dict() for o in active_orders}
        
        if "orders" not in data:
            data["orders"] = orders_dict
        else:
            data["orders"].update(orders_dict)
        
        data["order_update_timestamp"] = time.time()
        
        return data
    
    async def _cancel_old_orders(self, active_orders: List[Order]) -> None:
        """
        Cancelar órdenes que llevan demasiado tiempo activas.
        
        Args:
            active_orders: Lista de órdenes activas
        """
        now = time.time()
        
        for order in active_orders:
            if order.timestamp + self.auto_cancel_after < now:
                logger.info(f"Cancelando orden antigua: {order.id}, edad: {(now - order.timestamp) / 60:.1f} minutos")
                await self.order_manager.cancel_order(order.id)
    
    async def get_account_status(self) -> Dict[str, Any]:
        """
        Obtener estado de la cuenta.
        
        Returns:
            Estado de la cuenta
        """
        # Obtener exchange por defecto
        exchange_id = self.order_manager.default_exchange
        
        if not exchange_id or exchange_id not in self.order_manager.exchanges:
            logger.error(f"Exchange por defecto no válido: {exchange_id}")
            return {}
        
        exchange = self.order_manager.exchanges[exchange_id]
        
        # Obtener balance
        balance = await exchange.get_account_balance()
        
        # Obtener órdenes activas
        active_orders = await self.order_manager.get_active_orders()
        
        return {
            "balance": balance,
            "active_orders_count": len(active_orders),
            "exchange_id": exchange_id,
            "timestamp": time.time()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del motor de ejecución.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = super().get_stats()
        
        # Agregar estadísticas específicas
        engine_stats = {
            "simulation_mode": self.simulation_mode,
            "auto_cancel_after": self.auto_cancel_after,
            "order_manager_stats": self.order_manager.get_stats()
        }
        
        stats.update(engine_stats)
        return stats

# Función de ejecución para el pipeline
async def process_execution(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función de ejecución de órdenes para el pipeline.
    
    Args:
        data: Datos con decisiones
        context: Contexto de ejecución
        
    Returns:
        Datos con resultados de ejecución
    """
    engine = ExecutionEngine()
    
    # Inicializar si es necesario
    if not engine.order_manager.exchanges:
        await engine.initialize()
    
    # Ejecutar decisiones
    execution_data = await engine.execute_decisions(data)
    
    # Actualizar estados de órdenes
    updated_data = await engine.update_order_statuses(execution_data)
    
    # Obtener estado de la cuenta
    account_status = await engine.get_account_status()
    updated_data["account_status"] = account_status
    
    # Registrar información en el contexto
    context["execution_engine"] = engine.simulation_mode
    context["order_count"] = len(execution_data.get("execution_results", []))
    
    logger.info(f"Ejecución completada: {context['order_count']} órdenes procesadas")
    return updated_data

# Instancia global para uso directo
execution_engine = ExecutionEngine()