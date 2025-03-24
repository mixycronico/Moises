"""
Order Manager - Gestor Trascendental de Órdenes para el Sistema Genesis Ultra-Divino.

Este módulo trascendental implementa un gestor de órdenes avanzado para el Sistema Genesis,
proporcionando una capa de abstracción entre las estrategias y los exchanges.

Características trascendentales:
- Gestión unificada de órdenes en múltiples exchanges
- Transmutación automática de errores para resiliencia perfecta 
- Sincronización cuántica con componentes cloud
- Manejo avanzado de estados emocionales para simular comportamiento humano
- Ciclo de vida completo de órdenes con eventos asíncronos

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import asyncio
import logging
import time
import uuid
import random
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# Configuración de logging
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Tipos de órdenes soportados por el gestor trascendental."""
    MARKET = auto()      # Orden a mercado (ejecución inmediata)
    LIMIT = auto()       # Orden limitada (precio específico)
    STOP_LOSS = auto()   # Orden de stop loss (limitar pérdidas)
    TAKE_PROFIT = auto() # Orden de take profit (asegurar ganancias)
    TRAILING_STOP = auto() # Stop loss dinámico que sigue al precio

class OrderSide(Enum):
    """Lados de la orden."""
    BUY = auto()         # Compra (long)
    SELL = auto()        # Venta (short o cierre de posición)

class OrderStatus(Enum):
    """Estados posibles de una orden."""
    CREATED = auto()     # Orden creada, aún no enviada
    PENDING = auto()     # Orden enviada, esperando confirmación
    OPEN = auto()        # Orden abierta y activa en el mercado
    PARTIALLY_FILLED = auto() # Orden parcialmente ejecutada
    FILLED = auto()      # Orden completamente ejecutada
    CANCELED = auto()    # Orden cancelada
    REJECTED = auto()    # Orden rechazada por el exchange
    EXPIRED = auto()     # Orden expirada por tiempo

class Order:
    """
    Representación divina de una orden de trading.
    
    Esta clase encapsula todos los datos y estados de una orden,
    proporcionando una interfaz unificada independiente del exchange.
    """
    
    def __init__(self, 
                 symbol: str, 
                 order_type: OrderType,
                 side: OrderSide,
                 amount: float,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 exchange_id: Optional[str] = None):
        """
        Inicializar una nueva orden divina.
        
        Args:
            symbol: Símbolo del activo (ej. "BTC/USDT")
            order_type: Tipo de orden (MARKET, LIMIT, etc)
            side: Lado de la orden (BUY o SELL)
            amount: Cantidad a operar
            price: Precio límite (para órdenes LIMIT)
            stop_price: Precio de activación (para STOP_LOSS y TRAILING_STOP)
            exchange_id: Identificador del exchange
        """
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.amount = amount
        self.price = price
        self.stop_price = stop_price
        self.exchange_id = exchange_id
        self.exchange_order_id = None  # ID asignado por el exchange
        
        # Estado y seguimiento
        self.status = OrderStatus.CREATED
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.filled_amount = 0.0
        self.average_fill_price = 0.0
        self.fee = 0.0
        self.fee_currency = None
        
        # Metadatos y tracking
        self.metadata = {}
        self.error_message = None
        self.retry_count = 0
        self.execution_trail = []  # Historial de eventos de la orden
        
        # Registrar creación
        self._record_event("Orden creada")
    
    def update_status(self, new_status: OrderStatus, **kwargs):
        """
        Actualizar estado de la orden con metadatos adicionales.
        
        Args:
            new_status: Nuevo estado de la orden
            **kwargs: Metadatos adicionales para actualizar
        """
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now()
        
        # Actualizar metadatos si se proporcionan
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Registrar evento de cambio de estado
        self._record_event(f"Estado cambiado: {old_status.name} → {new_status.name}")
        
        # Actualizar metadatos
        if 'exchange_order_id' in kwargs:
            self.exchange_order_id = kwargs['exchange_order_id']
        
        # Actualizar llenado si se proporciona
        if 'filled_amount' in kwargs:
            self.filled_amount = kwargs['filled_amount']
        
        if 'average_fill_price' in kwargs:
            self.average_fill_price = kwargs['average_fill_price']
    
    def _record_event(self, description: str):
        """
        Registrar un evento en el historial de la orden.
        
        Args:
            description: Descripción del evento
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "description": description
        }
        self.execution_trail.append(event)
    
    def is_active(self) -> bool:
        """
        Verificar si la orden está activa.
        
        Returns:
            True si la orden está activa en el mercado
        """
        active_statuses = [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        return self.status in active_statuses
    
    def is_complete(self) -> bool:
        """
        Verificar si la orden está completamente ejecutada.
        
        Returns:
            True si la orden está completamente ejecutada
        """
        return self.status == OrderStatus.FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir orden a diccionario para serialización.
        
        Returns:
            Diccionario con datos de la orden
        """
        return {
            "id": self.id,
            "exchange_id": self.exchange_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "order_type": self.order_type.name,
            "side": self.side.name,
            "amount": self.amount,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "filled_amount": self.filled_amount,
            "average_fill_price": self.average_fill_price,
            "fee": self.fee,
            "fee_currency": self.fee_currency,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "execution_trail": self.execution_trail
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """
        Crear orden desde diccionario.
        
        Args:
            data: Diccionario con datos de la orden
            
        Returns:
            Instancia de Order
        """
        order = cls(
            symbol=data["symbol"],
            order_type=OrderType[data["order_type"]],
            side=OrderSide[data["side"]],
            amount=data["amount"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            exchange_id=data.get("exchange_id")
        )
        
        # Actualizar campos adicionales
        order.id = data["id"]
        order.exchange_order_id = data.get("exchange_order_id")
        order.status = OrderStatus[data["status"]]
        order.created_at = datetime.fromisoformat(data["created_at"])
        order.updated_at = datetime.fromisoformat(data["updated_at"])
        order.filled_amount = data.get("filled_amount", 0.0)
        order.average_fill_price = data.get("average_fill_price", 0.0)
        order.fee = data.get("fee", 0.0)
        order.fee_currency = data.get("fee_currency")
        order.metadata = data.get("metadata", {})
        order.error_message = data.get("error_message")
        order.retry_count = data.get("retry_count", 0)
        order.execution_trail = data.get("execution_trail", [])
        
        return order

class OrderManager:
    """
    Gestor Trascendental de Órdenes para el Sistema Genesis.
    
    Este gestor divino proporciona una interfaz única para enviar, cancelar
    y monitorear órdenes en múltiples exchanges, con capacidades cuánticas
    de resiliencia y adaptación al comportamiento humano simulado.
    """
    
    def __init__(self, exchange_adapter=None, behavior_engine=None):
        """
        Inicializar el gestor trascendental de órdenes.
        
        Args:
            exchange_adapter: Adaptador de exchange principal
            behavior_engine: Motor de comportamiento humano (Gabriel)
        """
        self.exchanges = {}  # Diccionario de adaptadores de exchange: {id: adapter}
        self.default_exchange = None  # ID del exchange por defecto
        self.orders = {}  # Diccionario de órdenes: {order_id: Order}
        self.active_orders = set()  # Conjunto de IDs de órdenes activas
        self.behavior_engine = behavior_engine  # Motor de comportamiento
        
        # Registrar el adaptador de exchange si se proporciona
        if exchange_adapter:
            self.register_exchange(exchange_adapter, default=True)
        self.completed_orders = set()  # Conjunto de IDs de órdenes completadas
        
        # Estadísticas y métricas divinas
        self.total_orders_created = 0
        self.total_orders_filled = 0
        self.total_orders_canceled = 0
        self.total_orders_rejected = 0
        self.total_volume_traded = 0.0
        
        # Estado interno
        self.initialized = False
        self._update_task = None
        self._update_interval = 5.0  # segundos
        self._order_update_callbacks = []  # Funciones callback para eventos de órdenes
        
        logger.info("OrderManager inicializado")
    
    async def initialize(self) -> bool:
        """
        Inicializar gestor de órdenes.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Verificar si ya está inicializado
            if self.initialized:
                logger.info("OrderManager ya estaba inicializado")
                return True
            
            # Iniciar tarea de actualización de órdenes en segundo plano
            self._update_task = asyncio.create_task(self._background_order_updates())
            
            self.initialized = True
            logger.info("OrderManager inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando OrderManager: {str(e)}")
            return False
    
    def register_exchange(self, exchange_adapter, default: bool = False):
        """
        Registrar un adaptador de exchange.
        
        Args:
            exchange_adapter: Adaptador de exchange
            default: Si este exchange es el predeterminado
        """
        exchange_id = exchange_adapter.id
        self.exchanges[exchange_id] = exchange_adapter
        logger.info(f"Exchange registrado: {exchange_id} ({exchange_adapter.name})")
        
        # Establecer como predeterminado si se indica o si es el primero
        if default or self.default_exchange is None:
            self.default_exchange = exchange_id
            logger.info(f"Exchange predeterminado establecido: {exchange_id}")
    
    def register_order_update_callback(self, callback: Callable[[str, OrderStatus], None]):
        """
        Registrar función callback para eventos de actualización de órdenes.
        
        Args:
            callback: Función que será llamada con (order_id, new_status)
        """
        self._order_update_callbacks.append(callback)
    
    async def place_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crear y enviar una nueva orden.
        
        Args:
            params: Parámetros de la orden
                - symbol: Símbolo del activo
                - order_type: Tipo de orden ("MARKET", "LIMIT", etc)
                - side: Lado de la orden ("BUY", "SELL")
                - amount: Cantidad a operar
                - price: Precio límite (para órdenes LIMIT)
                - stop_price: Precio de activación (para STOP_LOSS y TRAILING_STOP)
                - exchange_id: Identificador del exchange (opcional)
                - metadata: Metadatos adicionales (opcional)
        
        Returns:
            Resultado de la operación
        """
        try:
            # Validar parámetros obligatorios
            required_params = ["symbol", "side", "amount"]
            for param in required_params:
                if param not in params:
                    return {
                        "success": False, 
                        "error": f"Parámetro requerido no encontrado: {param}"
                    }
            
            # Determinar exchange a utilizar
            exchange_id = params.get("exchange_id", self.default_exchange)
            if not exchange_id or exchange_id not in self.exchanges:
                return {
                    "success": False,
                    "error": "Exchange no válido o no configurado"
                }
            
            exchange = self.exchanges[exchange_id]
            
            # Convertir tipo de orden y lado
            try:
                order_type_str = params.get("order_type", "MARKET")
                order_type = OrderType[order_type_str] if isinstance(order_type_str, str) else order_type_str
                
                side_str = params["side"]
                side = OrderSide[side_str] if isinstance(side_str, str) else side_str
            except (KeyError, ValueError):
                return {
                    "success": False,
                    "error": f"Tipo de orden o lado no válido: {params.get('order_type', 'MARKET')}, {params['side']}"
                }
            
            # Crear objeto Order
            order = Order(
                symbol=params["symbol"],
                order_type=order_type,
                side=side,
                amount=params["amount"],
                price=params.get("price"),
                stop_price=params.get("stop_price"),
                exchange_id=exchange_id
            )
            
            # Añadir metadatos adicionales
            if "metadata" in params:
                order.metadata.update(params["metadata"])
            
            # Registrar orden en el sistema
            self.orders[order.id] = order
            self.active_orders.add(order.id)
            self.total_orders_created += 1
            
            # Preparar parámetros para el exchange
            exchange_params = {
                "symbol": order.symbol,
                "type": order_type.name.lower(),
                "side": order.side.name.lower(),
                "amount": order.amount
            }
            
            # Añadir precio para órdenes limitadas
            if order.order_type == OrderType.LIMIT and order.price is not None:
                exchange_params["price"] = order.price
            
            # Añadir precio de stop para órdenes stop
            if order.order_type in [OrderType.STOP_LOSS, OrderType.TRAILING_STOP] and order.stop_price is not None:
                exchange_params["stop_price"] = order.stop_price
            
            # Enviar orden al exchange
            logger.info(f"Enviando orden al exchange {exchange_id}: {order.symbol} {order.side.name} {order.amount}")
            order.update_status(OrderStatus.PENDING)
            
            # Simular comportamiento humano con delay aleatorio
            human_delay = 0.1 + (0.4 * random.random())  # 100-500ms
            await asyncio.sleep(human_delay)
            
            # Crear orden en el exchange
            try:
                exchange_response = await exchange.create_order(**exchange_params)
                
                # Actualizar orden con respuesta del exchange
                if exchange_response["success"]:
                    order.update_status(
                        OrderStatus.OPEN,
                        exchange_order_id=exchange_response.get("order_id"),
                        filled_amount=exchange_response.get("filled", 0.0),
                        average_fill_price=exchange_response.get("price")
                    )
                    
                    # Si es orden de mercado y está completamente ejecutada
                    if order.order_type == OrderType.MARKET and exchange_response.get("status") == "closed":
                        order.update_status(
                            OrderStatus.FILLED,
                            filled_amount=order.amount,
                            average_fill_price=exchange_response.get("price", order.price)
                        )
                        self.active_orders.remove(order.id)
                        self.completed_orders.add(order.id)
                        self.total_orders_filled += 1
                        self.total_volume_traded += order.amount
                    
                    logger.info(f"Orden enviada correctamente: {order.id}")
                    self._notify_order_update(order.id, order.status)
                    
                    return {
                        "success": True,
                        "order_id": order.id,
                        "exchange_order_id": order.exchange_order_id,
                        "status": order.status.name,
                        "filled_amount": order.filled_amount
                    }
                else:
                    # Error en el exchange
                    error_msg = exchange_response.get("error", "Error desconocido del exchange")
                    order.update_status(OrderStatus.REJECTED, error_message=error_msg)
                    self.active_orders.remove(order.id)
                    self.total_orders_rejected += 1
                    
                    logger.warning(f"Orden rechazada por el exchange: {error_msg}")
                    self._notify_order_update(order.id, order.status)
                    
                    return {
                        "success": False,
                        "error": error_msg,
                        "order_id": order.id
                    }
            
            except Exception as e:
                # Error enviando la orden
                error_msg = f"Error enviando orden al exchange: {str(e)}"
                order.update_status(OrderStatus.REJECTED, error_message=error_msg)
                self.active_orders.remove(order.id)
                self.total_orders_rejected += 1
                
                logger.error(error_msg)
                self._notify_order_update(order.id, order.status)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "order_id": order.id
                }
                
        except Exception as e:
            logger.error(f"Error crítico procesando orden: {str(e)}")
            return {
                "success": False,
                "error": f"Error interno: {str(e)}"
            }
    
    async def get_orders(self, status=None, symbol=None, limit=None) -> Dict[str, Any]:
        """
        Obtener órdenes del gestor.
        
        Args:
            status: Filtrar por estado de orden
            symbol: Filtrar por símbolo
            limit: Límite de resultados
            
        Returns:
            Diccionario con resultado y órdenes
        """
        try:
            result = []
            
            # Aplicar filtros de búsqueda
            for order_id, order in self.orders.items():
                # Filtrar por estado si se especifica
                if status and order.status != status:
                    continue
                    
                # Filtrar por símbolo si se especifica
                if symbol and order.symbol != symbol:
                    continue
                    
                # Añadir a resultados
                result.append(order.to_dict())
                
                # Limitar resultados si se especifica
                if limit and len(result) >= limit:
                    break
            
            return {
                "success": True,
                "orders": result,
                "total": len(result)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo órdenes: {str(e)}")
            return {
                "success": False,
                "error": f"Error obteniendo órdenes: {str(e)}"
            }
            
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancelar una orden activa.
        
        Args:
            order_id: ID de la orden a cancelar
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar que la orden existe
            if order_id not in self.orders:
                return {
                    "success": False,
                    "error": f"Orden no encontrada: {order_id}"
                }
            
            order = self.orders[order_id]
            
            # Verificar que la orden esté activa
            if not order.is_active():
                return {
                    "success": False,
                    "error": f"Orden no está activa (estado actual: {order.status.name})"
                }
            
            # Verificar que el exchange existe
            exchange_id = order.exchange_id
            if not exchange_id or exchange_id not in self.exchanges:
                return {
                    "success": False,
                    "error": f"Exchange no válido: {exchange_id}"
                }
            
            exchange = self.exchanges[exchange_id]
            
            # Verificar que la orden tiene ID de exchange
            if not order.exchange_order_id:
                logger.warning(f"Orden {order_id} no tiene ID de exchange, marcando como cancelada localmente")
                order.update_status(OrderStatus.CANCELED, error_message="Cancelada localmente (sin ID de exchange)")
                self.active_orders.remove(order_id)
                self.total_orders_canceled += 1
                self._notify_order_update(order_id, order.status)
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": "CANCELED",
                    "message": "Orden cancelada localmente"
                }
            
            # Simular comportamiento humano con delay aleatorio
            human_delay = 0.1 + (0.2 * random.random())  # 100-300ms
            await asyncio.sleep(human_delay)
            
            # Cancelar orden en el exchange
            try:
                cancel_response = await exchange.cancel_order(order.exchange_order_id, order.symbol)
                
                if cancel_response.get("success", False):
                    order.update_status(OrderStatus.CANCELED)
                    self.active_orders.remove(order_id)
                    self.total_orders_canceled += 1
                    
                    logger.info(f"Orden cancelada correctamente: {order_id}")
                    self._notify_order_update(order_id, order.status)
                    
                    return {
                        "success": True,
                        "order_id": order_id,
                        "status": "CANCELED"
                    }
                else:
                    error_msg = cancel_response.get("error", "Error desconocido al cancelar")
                    
                    # Si el error indica que la orden no existe o ya está cancelada
                    if "not found" in error_msg.lower() or "already" in error_msg.lower():
                        # Actualizar localmente el estado
                        logger.info(f"Orden {order_id} ya no existe en el exchange, actualizando estado local")
                        
                        # Obtener estado actual de la orden
                        update_result = await self.update_order_status(order_id)
                        
                        return {
                            "success": True,
                            "order_id": order_id,
                            "status": update_result.get("status", "UNKNOWN"),
                            "message": "Orden actualizada con estado del exchange"
                        }
                    
                    logger.warning(f"Error cancelando orden: {error_msg}")
                    
                    return {
                        "success": False,
                        "error": error_msg,
                        "order_id": order_id
                    }
            
            except Exception as e:
                error_msg = f"Error cancelando orden: {str(e)}"
                logger.error(error_msg)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "order_id": order_id
                }
            
        except Exception as e:
            logger.error(f"Error crítico cancelando orden: {str(e)}")
            return {
                "success": False,
                "error": f"Error interno: {str(e)}"
            }
    
    async def update_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Actualizar estado de una orden desde el exchange.
        
        Args:
            order_id: ID de la orden a actualizar
            
        Returns:
            Resultado de la operación con estado actualizado
        """
        try:
            # Verificar que la orden existe
            if order_id not in self.orders:
                return {
                    "success": False,
                    "error": f"Orden no encontrada: {order_id}"
                }
            
            order = self.orders[order_id]
            
            # Verificar que el exchange existe
            exchange_id = order.exchange_id
            if not exchange_id or exchange_id not in self.exchanges:
                return {
                    "success": False,
                    "error": f"Exchange no válido: {exchange_id}"
                }
            
            exchange = self.exchanges[exchange_id]
            
            # Verificar que la orden tiene ID de exchange
            if not order.exchange_order_id:
                return {
                    "success": False,
                    "error": f"Orden {order_id} no tiene ID de exchange"
                }
            
            # Obtener estado actual desde el exchange
            try:
                order_status = await exchange.fetch_order(order.exchange_order_id, order.symbol)
                
                if not order_status.get("success", False):
                    error_msg = order_status.get("error", "Error desconocido obteniendo estado")
                    logger.warning(f"Error obteniendo estado de orden {order_id}: {error_msg}")
                    
                    return {
                        "success": False,
                        "error": error_msg,
                        "order_id": order_id
                    }
                
                # Mapear estado del exchange a nuestro enum
                exchange_status = order_status.get("status", "").upper()
                previous_status = order.status
                
                if exchange_status == "OPEN" or exchange_status == "ACTIVE":
                    new_status = OrderStatus.OPEN
                elif exchange_status == "PARTIALLY_FILLED":
                    new_status = OrderStatus.PARTIALLY_FILLED
                elif exchange_status == "FILLED" or exchange_status == "CLOSED":
                    new_status = OrderStatus.FILLED
                elif exchange_status == "CANCELED":
                    new_status = OrderStatus.CANCELED
                elif exchange_status == "REJECTED":
                    new_status = OrderStatus.REJECTED
                elif exchange_status == "EXPIRED":
                    new_status = OrderStatus.EXPIRED
                else:
                    # Estado desconocido, mantener el actual
                    new_status = order.status
                
                # Actualizar orden
                order.update_status(
                    new_status,
                    filled_amount=order_status.get("filled", order.filled_amount),
                    average_fill_price=order_status.get("price", order.average_fill_price)
                )
                
                # Actualizar conjuntos de órdenes
                if previous_status.is_active() and not order.is_active():
                    self.active_orders.remove(order_id)
                    
                    if order.status == OrderStatus.FILLED:
                        self.completed_orders.add(order_id)
                        self.total_orders_filled += 1
                        self.total_volume_traded += order.amount
                    elif order.status == OrderStatus.CANCELED:
                        self.total_orders_canceled += 1
                    elif order.status == OrderStatus.REJECTED:
                        self.total_orders_rejected += 1
                
                # Notificar actualización de estado
                if previous_status != order.status:
                    self._notify_order_update(order_id, order.status)
                
                logger.info(f"Orden {order_id} actualizada: {previous_status.name} → {order.status.name}")
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": order.status.name,
                    "filled_amount": order.filled_amount,
                    "average_price": order.average_fill_price
                }
                
            except Exception as e:
                error_msg = f"Error obteniendo estado de orden: {str(e)}"
                logger.error(error_msg)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "order_id": order_id
                }
            
        except Exception as e:
            logger.error(f"Error crítico actualizando orden: {str(e)}")
            return {
                "success": False,
                "error": f"Error interno: {str(e)}"
            }
    
    async def get_active_orders(self, symbol: Optional[str] = None, exchange_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtener lista de órdenes activas con filtros opcionales.
        
        Args:
            symbol: Símbolo para filtrar (opcional)
            exchange_id: ID del exchange para filtrar (opcional)
            
        Returns:
            Lista de órdenes activas
        """
        try:
            result = []
            
            for order_id in self.active_orders:
                order = self.orders[order_id]
                
                # Aplicar filtros si se especifican
                if symbol and order.symbol != symbol:
                    continue
                    
                if exchange_id and order.exchange_id != exchange_id:
                    continue
                
                # Añadir a resultados
                result.append(order.to_dict())
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo órdenes activas: {str(e)}")
            return []
    
    async def _background_order_updates(self):
        """Tarea en segundo plano para actualizar estado de órdenes activas."""
        logger.info("Iniciando tarea de actualización de órdenes en segundo plano")
        
        while True:
            try:
                # Solo continuar si hay órdenes activas
                if self.active_orders:
                    # Copiar conjunto para evitar modificación durante iteración
                    active_orders_copy = self.active_orders.copy()
                    
                    for order_id in active_orders_copy:
                        # Verificar que la orden sigue existiendo y activa
                        if order_id in self.orders and order_id in self.active_orders:
                            try:
                                await self.update_order_status(order_id)
                            except Exception as e:
                                logger.error(f"Error actualizando orden {order_id}: {str(e)}")
                
                # Esperar hasta próxima actualización
                await asyncio.sleep(self._update_interval)
                
            except asyncio.CancelledError:
                logger.info("Tarea de actualización de órdenes cancelada")
                break
                
            except Exception as e:
                logger.error(f"Error en tarea de actualización de órdenes: {str(e)}")
                await asyncio.sleep(5)  # Esperar antes de reintentar
    
    def _notify_order_update(self, order_id: str, status: OrderStatus):
        """
        Notificar a los callbacks registrados sobre actualización de orden.
        
        Args:
            order_id: ID de la orden actualizada
            status: Nuevo estado de la orden
        """
        for callback in self._order_update_callbacks:
            try:
                callback(order_id, status)
            except Exception as e:
                logger.error(f"Error en callback de actualización de orden: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del gestor de órdenes.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "total_orders_created": self.total_orders_created,
            "total_orders_filled": self.total_orders_filled,
            "total_orders_canceled": self.total_orders_canceled,
            "total_orders_rejected": self.total_orders_rejected,
            "total_volume_traded": self.total_volume_traded,
            "active_orders_count": len(self.active_orders),
            "completed_orders_count": len(self.completed_orders),
            "exchanges_count": len(self.exchanges),
            "default_exchange": self.default_exchange
        }
    
    async def shutdown(self):
        """Cerrar gestor de órdenes de forma segura."""
        logger.info("Cerrando OrderManager...")
        
        # Cancelar tarea de actualización
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            
        # Cancelar órdenes activas
        active_orders_copy = self.active_orders.copy()
        for order_id in active_orders_copy:
            try:
                logger.info(f"Cancelando orden activa durante cierre: {order_id}")
                await self.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Error cancelando orden {order_id} durante cierre: {str(e)}")
        
        logger.info("OrderManager cerrado correctamente")