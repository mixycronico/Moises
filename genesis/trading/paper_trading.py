"""
Sistema de Paper Trading para Genesis.

Este módulo proporciona las funcionalidades para ejecutar operaciones simuladas
con datos reales, permitiendo probar estrategias sin riesgo financiero.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from sqlalchemy import create_engine, select, and_, func, desc, or_
from sqlalchemy.orm import Session, sessionmaker

from genesis.core.base import Component
from genesis.db.paper_trading_models import (
    PaperTradingAccount, PaperTradingBalance, PaperTradingOrder,
    PaperTradingTrade, MarketData
)

class PaperTradingManager(Component):
    """
    Gestor del sistema de Paper Trading.
    
    Este componente maneja las cuentas de paper trading y ejecuta
    órdenes simuladas basadas en datos reales del mercado.
    """
    
    def __init__(self, name: str = "paper_trading_manager"):
        """
        Inicializar el gestor de paper trading.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        
        self.db_url = None
        self.engine = None
        self.Session = None
        
        self.accounts: Dict[int, Dict[str, Any]] = {}  # Cache de cuentas por ID
        self.current_prices: Dict[str, float] = {}  # Precios actuales por símbolo
        self.order_book_cache: Dict[str, Dict[str, List[List[float]]]] = {}  # Libro de órdenes por símbolo
        
        self.update_interval = 1.0  # Segundos
        self.processing_task = None
        
        self.logger = logging.getLogger(f"paper_trading.{name}")
    
    async def start(self) -> None:
        """Iniciar el gestor de paper trading."""
        await super().start()
        
        # Inicializar la conexión a la base de datos
        import os
        self.db_url = os.environ.get('DATABASE_URL')
        if not self.db_url:
            self.logger.error("No se encontró la variable de entorno DATABASE_URL")
            return
        
        try:
            self.engine = create_engine(self.db_url)
            self.Session = sessionmaker(bind=self.engine)
            
            # Cargar cuentas activas
            await self._load_accounts()
            
            # Iniciar tarea de procesamiento
            self.processing_task = asyncio.create_task(self._process_orders_loop())
            
            self.logger.info("Sistema de Paper Trading iniciado correctamente")
            
            # Emitir evento de componente listo
            await self.emit_event("paper_trading.ready", {
                "status": "ready",
                "accounts_loaded": len(self.accounts)
            })
            
        except Exception as e:
            self.logger.error(f"Error al iniciar el sistema de Paper Trading: {e}")
            # Emitir evento de error
            await self.emit_event("paper_trading.error", {
                "error": str(e)
            })
    
    async def stop(self) -> None:
        """Detener el gestor de paper trading."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Cerrar el engine de la base de datos
        if self.engine:
            self.engine.dispose()
        
        self.logger.info("Sistema de Paper Trading detenido")
        await super().stop()
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que envió el evento
        """
        if event_type == "market.ticker":
            # Actualizar precios con datos de mercado
            symbol = data.get("symbol")
            ticker = data.get("ticker")
            if symbol and ticker and "last" in ticker:
                self.current_prices[symbol] = ticker["last"]
                self.logger.debug(f"Precio actualizado para {symbol}: {ticker['last']}")
        
        elif event_type == "market.order_book":
            # Actualizar cache del libro de órdenes
            symbol = data.get("symbol")
            order_book = data.get("order_book")
            if symbol and order_book:
                self.order_book_cache[symbol] = order_book
                self.logger.debug(f"Order book actualizado para {symbol}")
        
        elif event_type == "trading.order_request":
            # Procesar solicitud de nueva orden
            await self._handle_order_request(data)
        
        elif event_type == "trading.cancel_order_request":
            # Procesar solicitud de cancelación de orden
            await self._handle_cancel_request(data)
        
        elif event_type == "system.db_updated":
            # Recargar datos de cuentas si la base de datos ha cambiado
            if data.get("target") == "paper_trading":
                await self._load_accounts()
    
    async def _load_accounts(self) -> None:
        """Cargar cuentas activas de paper trading desde la base de datos."""
        try:
            session = self.Session()
            accounts = session.query(PaperTradingAccount).filter_by(is_active=True).all()
            
            # Actualizar cache de cuentas
            self.accounts = {}
            for account in accounts:
                balances = {}
                for balance in account.balances:
                    balances[balance.asset] = {
                        "free": balance.free,
                        "locked": balance.locked,
                        "total": balance.free + balance.locked
                    }
                
                self.accounts[account.id] = {
                    "id": account.id,
                    "name": account.name,
                    "user_id": account.user_id,
                    "initial_balance": account.initial_balance_usd,
                    "balances": balances,
                    "last_updated": datetime.utcnow()
                }
            
            session.close()
            self.logger.info(f"Cargadas {len(self.accounts)} cuentas activas de Paper Trading")
            
        except Exception as e:
            self.logger.error(f"Error al cargar cuentas de Paper Trading: {e}")
    
    async def _process_orders_loop(self) -> None:
        """Loop continuo para procesar órdenes pendientes."""
        self.logger.info("Iniciando loop de procesamiento de órdenes")
        
        while True:
            try:
                # Procesar órdenes límite que puedan ejecutarse
                await self._process_limit_orders()
                
                # Actualizar precios de mercado si es necesario
                await self._update_market_prices()
                
                # Esperar hasta el próximo ciclo
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                self.logger.info("Loop de procesamiento de órdenes cancelado")
                break
            except Exception as e:
                self.logger.error(f"Error en el procesamiento de órdenes: {e}")
                await asyncio.sleep(5.0)  # Esperar un poco más en caso de error
    
    async def _update_market_prices(self) -> None:
        """Actualizar precios de mercado desde los datos históricos."""
        try:
            # En un sistema real, esto se haría con datos en tiempo real
            # Para paper trading, podemos usar los datos históricos más recientes
            session = self.Session()
            
            # Obtener los símbolos para los que tenemos órdenes activas
            symbols_query = session.query(PaperTradingOrder.symbol).filter(
                PaperTradingOrder.status.in_(['open', 'pending'])
            ).distinct()
            
            active_symbols = [row[0] for row in symbols_query]
            
            for symbol in active_symbols:
                # Obtener el precio más reciente de la base de datos
                latest_data = session.query(MarketData).filter(
                    MarketData.symbol == symbol
                ).order_by(desc(MarketData.timestamp)).first()
                
                if latest_data:
                    self.current_prices[symbol] = latest_data.close
                    
                    # Emitir evento de ticker simulado
                    await self.emit_event("market.ticker", {
                        "exchange_id": "paper_trading",
                        "symbol": symbol,
                        "ticker": {
                            "symbol": symbol,
                            "last": latest_data.close,
                            "bid": latest_data.close * 0.9995,  # Simulación simple
                            "ask": latest_data.close * 1.0005,  # Simulación simple
                            "high": latest_data.high,
                            "low": latest_data.low,
                            "open": latest_data.open,
                            "close": latest_data.close,
                            "volume": latest_data.volume,
                            "timestamp": int(latest_data.timestamp.timestamp() * 1000),
                            "datetime": latest_data.timestamp.isoformat(),
                            "source": "paper_trading"
                        }
                    })
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error al actualizar precios de mercado: {e}")
    
    async def _process_limit_orders(self) -> None:
        """Procesar órdenes límite que puedan ejecutarse con los precios actuales."""
        if not self.current_prices:
            return  # No hay precios para procesar órdenes
        
        try:
            session = self.Session()
            
            # Obtener órdenes límite abiertas
            open_orders = session.query(PaperTradingOrder).filter(
                and_(
                    PaperTradingOrder.status == 'open',
                    PaperTradingOrder.type == 'limit'
                )
            ).all()
            
            for order in open_orders:
                if order.symbol not in self.current_prices:
                    continue
                
                current_price = self.current_prices[order.symbol]
                
                # Verificar si la orden se puede ejecutar
                can_execute = False
                execution_price = order.price
                
                if order.side == 'buy' and current_price <= order.price:
                    can_execute = True
                    # Para compras, ejecutamos al precio de la orden o mejor
                    execution_price = min(current_price, order.price)
                
                elif order.side == 'sell' and current_price >= order.price:
                    can_execute = True
                    # Para ventas, ejecutamos al precio de la orden o mejor
                    execution_price = max(current_price, order.price)
                
                if can_execute:
                    # Ejecutar la orden
                    await self._execute_order(session, order, execution_price)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error al procesar órdenes límite: {e}")
    
    async def _handle_order_request(self, data: Dict[str, Any]) -> None:
        """
        Manejar solicitud de nueva orden.
        
        Args:
            data: Datos de la solicitud
        """
        account_id = data.get("account_id")
        symbol = data.get("symbol")
        side = data.get("side")
        order_type = data.get("type")
        quantity = data.get("quantity")
        price = data.get("price")
        stop_price = data.get("stop_price")
        
        if not all([account_id, symbol, side, order_type, quantity]):
            self.logger.error(f"Datos insuficientes para crear orden: {data}")
            await self.emit_event("trading.order_error", {
                "account_id": account_id,
                "error": "Datos insuficientes para crear orden",
                "data": data
            })
            return
        
        # Validar el tipo de orden
        if order_type not in ['market', 'limit', 'stop_loss', 'take_profit']:
            self.logger.error(f"Tipo de orden no válido: {order_type}")
            await self.emit_event("trading.order_error", {
                "account_id": account_id,
                "error": f"Tipo de orden no válido: {order_type}",
                "data": data
            })
            return
        
        # Validar precio para órdenes que lo requieren
        if order_type in ['limit', 'stop_loss', 'take_profit'] and price is None:
            self.logger.error(f"Precio requerido para orden {order_type}")
            await self.emit_event("trading.order_error", {
                "account_id": account_id,
                "error": f"Precio requerido para orden {order_type}",
                "data": data
            })
            return
        
        try:
            session = self.Session()
            
            # Verificar que la cuenta existe
            account = session.query(PaperTradingAccount).filter_by(id=account_id).first()
            if not account:
                session.close()
                self.logger.error(f"Cuenta no encontrada: {account_id}")
                await self.emit_event("trading.order_error", {
                    "account_id": account_id,
                    "error": f"Cuenta no encontrada: {account_id}",
                    "data": data
                })
                return
            
            # Extraer activos del símbolo (ej: BTC/USDT -> BTC y USDT)
            base_asset, quote_asset = symbol.split('/')
            
            # Verificar saldo suficiente
            if side == 'buy':
                required_asset = quote_asset
                required_amount = quantity * (price or self.current_prices.get(symbol, 0))
            else:  # sell
                required_asset = base_asset
                required_amount = quantity
            
            # Obtener balance
            balance = session.query(PaperTradingBalance).filter(
                and_(
                    PaperTradingBalance.account_id == account_id,
                    PaperTradingBalance.asset == required_asset
                )
            ).first()
            
            if not balance or balance.free < required_amount:
                session.close()
                self.logger.error(f"Saldo insuficiente de {required_asset} para orden: {data}")
                await self.emit_event("trading.order_error", {
                    "account_id": account_id,
                    "error": f"Saldo insuficiente de {required_asset}",
                    "data": data
                })
                return
            
            # Crear la orden
            order = PaperTradingOrder(
                account_id=account_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status='open',
                filled_quantity=0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            session.add(order)
            session.flush()  # Para obtener el ID asignado
            
            # Actualizar saldo (bloquear fondos)
            balance.free -= required_amount
            balance.locked += required_amount
            
            session.commit()
            
            order_response = {
                "order_id": order.order_id,
                "account_id": account_id,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "price": price,
                "status": "open",
                "created_at": order.created_at.isoformat()
            }
            
            # Emitir evento de orden creada
            await self.emit_event("trading.order_created", order_response)
            
            # Para órdenes de mercado, ejecutarlas inmediatamente
            if order_type == 'market':
                # Determinar precio de ejecución
                execution_price = self.current_prices.get(symbol)
                if not execution_price:
                    # Si no tenemos precio actual, usar datos históricos recientes
                    latest_data = session.query(MarketData).filter(
                        MarketData.symbol == symbol
                    ).order_by(desc(MarketData.timestamp)).first()
                    
                    if latest_data:
                        execution_price = latest_data.close
                    else:
                        execution_price = price or 0  # Último recurso
                
                await self._execute_order(session, order, execution_price)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error al crear orden: {e}")
            await self.emit_event("trading.order_error", {
                "account_id": account_id,
                "error": f"Error al crear orden: {e}",
                "data": data
            })
    
    async def _handle_cancel_request(self, data: Dict[str, Any]) -> None:
        """
        Manejar solicitud de cancelación de orden.
        
        Args:
            data: Datos de la solicitud
        """
        order_id = data.get("order_id")
        account_id = data.get("account_id")
        
        if not order_id:
            self.logger.error("ID de orden requerido para cancelación")
            await self.emit_event("trading.cancel_error", {
                "error": "ID de orden requerido para cancelación",
                "data": data
            })
            return
        
        try:
            session = self.Session()
            
            # Buscar la orden
            order = session.query(PaperTradingOrder).filter_by(order_id=order_id).first()
            
            if not order:
                session.close()
                self.logger.error(f"Orden no encontrada: {order_id}")
                await self.emit_event("trading.cancel_error", {
                    "order_id": order_id,
                    "error": f"Orden no encontrada: {order_id}",
                    "data": data
                })
                return
            
            # Verificar que la orden puede ser cancelada
            if order.status not in ['open', 'pending']:
                session.close()
                self.logger.error(f"No se puede cancelar orden con estado: {order.status}")
                await self.emit_event("trading.cancel_error", {
                    "order_id": order_id,
                    "error": f"No se puede cancelar orden con estado: {order.status}",
                    "data": data
                })
                return
            
            # Extraer activos del símbolo
            base_asset, quote_asset = order.symbol.split('/')
            
            # Desbloquear fondos
            if order.side == 'buy':
                required_asset = quote_asset
                remaining_amount = (order.quantity - order.filled_quantity) * order.price
            else:  # sell
                required_asset = base_asset
                remaining_amount = order.quantity - order.filled_quantity
            
            # Actualizar balance
            balance = session.query(PaperTradingBalance).filter(
                and_(
                    PaperTradingBalance.account_id == order.account_id,
                    PaperTradingBalance.asset == required_asset
                )
            ).first()
            
            if balance:
                balance.locked -= remaining_amount
                balance.free += remaining_amount
            
            # Actualizar estado de la orden
            order.status = 'canceled'
            order.updated_at = datetime.utcnow()
            
            session.commit()
            
            # Emitir evento de orden cancelada
            await self.emit_event("trading.order_canceled", {
                "order_id": order.order_id,
                "account_id": order.account_id,
                "symbol": order.symbol,
                "status": "canceled",
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.quantity - order.filled_quantity
            })
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error al cancelar orden: {e}")
            await self.emit_event("trading.cancel_error", {
                "order_id": order_id,
                "error": f"Error al cancelar orden: {e}",
                "data": data
            })
    
    async def _execute_order(self, session: Session, order: PaperTradingOrder, execution_price: float) -> None:
        """
        Ejecutar una orden.
        
        Args:
            session: Sesión de base de datos
            order: Orden a ejecutar
            execution_price: Precio de ejecución
        """
        try:
            # Extraer activos del símbolo
            base_asset, quote_asset = order.symbol.split('/')
            
            # Calcular cantidad a ejecutar (podría ser parcial)
            remaining_quantity = order.quantity - order.filled_quantity
            execution_quantity = remaining_quantity  # Para simplificar, ejecutamos todo de una vez
            
            # Calcular comisión (simulado)
            commission_rate = 0.001  # 0.1%
            commission_amount = execution_quantity * execution_price * commission_rate
            commission_asset = quote_asset
            
            # Crear registro de operación (trade)
            trade = PaperTradingTrade(
                order_id=order.order_id,
                account_id=order.account_id,
                symbol=order.symbol,
                side=order.side,
                quantity=execution_quantity,
                price=execution_price,
                commission=commission_amount,
                commission_asset=commission_asset,
                timestamp=datetime.utcnow()
            )
            
            session.add(trade)
            
            # Actualizar la orden
            order.filled_quantity += execution_quantity
            order.average_price = (
                (order.average_price or 0) * (order.filled_quantity - execution_quantity) +
                execution_price * execution_quantity
            ) / order.filled_quantity if order.filled_quantity > 0 else 0
            
            if order.filled_quantity >= order.quantity:
                order.status = 'closed'
            
            order.updated_at = datetime.utcnow()
            
            # Actualizar saldos
            if order.side == 'buy':
                # Compra: gastar quote_asset, recibir base_asset
                quote_balance = session.query(PaperTradingBalance).filter(
                    and_(
                        PaperTradingBalance.account_id == order.account_id,
                        PaperTradingBalance.asset == quote_asset
                    )
                ).first()
                
                base_balance = session.query(PaperTradingBalance).filter(
                    and_(
                        PaperTradingBalance.account_id == order.account_id,
                        PaperTradingBalance.asset == base_asset
                    )
                ).first()
                
                # Crear balance de base_asset si no existe
                if not base_balance:
                    base_balance = PaperTradingBalance(
                        account_id=order.account_id,
                        asset=base_asset,
                        free=0.0,
                        locked=0.0
                    )
                    session.add(base_balance)
                
                # Actualizar saldos
                total_cost = execution_quantity * execution_price
                quote_balance.locked -= total_cost
                base_balance.free += execution_quantity - (commission_amount if commission_asset == base_asset else 0)
                
                # Pagar comisión
                if commission_asset == quote_asset:
                    quote_balance.locked -= commission_amount
                
            else:  # sell
                # Venta: gastar base_asset, recibir quote_asset
                base_balance = session.query(PaperTradingBalance).filter(
                    and_(
                        PaperTradingBalance.account_id == order.account_id,
                        PaperTradingBalance.asset == base_asset
                    )
                ).first()
                
                quote_balance = session.query(PaperTradingBalance).filter(
                    and_(
                        PaperTradingBalance.account_id == order.account_id,
                        PaperTradingBalance.asset == quote_asset
                    )
                ).first()
                
                # Crear balance de quote_asset si no existe
                if not quote_balance:
                    quote_balance = PaperTradingBalance(
                        account_id=order.account_id,
                        asset=quote_asset,
                        free=0.0,
                        locked=0.0
                    )
                    session.add(quote_balance)
                
                # Actualizar saldos
                total_received = execution_quantity * execution_price
                base_balance.locked -= execution_quantity
                quote_balance.free += total_received - (commission_amount if commission_asset == quote_asset else 0)
                
                # Pagar comisión
                if commission_asset == base_asset:
                    base_balance.locked -= commission_amount
            
            session.commit()
            
            # Emitir evento de operación ejecutada
            trade_response = {
                "trade_id": trade.trade_id,
                "order_id": order.order_id,
                "account_id": order.account_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": execution_quantity,
                "price": execution_price,
                "commission": commission_amount,
                "commission_asset": commission_asset,
                "timestamp": trade.timestamp.isoformat()
            }
            
            asyncio.create_task(self.emit_event("trading.trade_executed", trade_response))
            
            # Si la orden se completó, emitir evento
            if order.status == 'closed':
                order_response = {
                    "order_id": order.order_id,
                    "account_id": order.account_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.type,
                    "quantity": order.quantity,
                    "price": order.price,
                    "average_price": order.average_price,
                    "status": "closed",
                    "filled_quantity": order.filled_quantity,
                    "updated_at": order.updated_at.isoformat()
                }
                
                asyncio.create_task(self.emit_event("trading.order_closed", order_response))
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error al ejecutar orden {order.order_id}: {e}")
            
            # Emitir evento de error
            asyncio.create_task(self.emit_event("trading.trade_error", {
                "order_id": order.order_id,
                "account_id": order.account_id,
                "error": f"Error al ejecutar orden: {e}"
            }))
    
    async def get_account_balance(self, account_id: int) -> Dict[str, Any]:
        """
        Obtener saldo de cuenta.
        
        Args:
            account_id: ID de la cuenta
            
        Returns:
            Saldo de la cuenta por activo
        """
        try:
            session = self.Session()
            
            balances = session.query(PaperTradingBalance).filter_by(account_id=account_id).all()
            
            result = {}
            for balance in balances:
                result[balance.asset] = {
                    "free": balance.free,
                    "locked": balance.locked,
                    "total": balance.free + balance.locked
                }
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener saldo: {e}")
            return {}
    
    async def get_open_orders(self, account_id: int, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtener órdenes abiertas.
        
        Args:
            account_id: ID de la cuenta
            symbol: Símbolo de trading (opcional)
            
        Returns:
            Lista de órdenes abiertas
        """
        try:
            session = self.Session()
            
            query = session.query(PaperTradingOrder).filter(
                and_(
                    PaperTradingOrder.account_id == account_id,
                    PaperTradingOrder.status.in_(["open", "pending"])
                )
            )
            
            if symbol:
                query = query.filter(PaperTradingOrder.symbol == symbol)
            
            orders = query.all()
            
            result = []
            for order in orders:
                result.append({
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.type,
                    "quantity": order.quantity,
                    "price": order.price,
                    "stop_price": order.stop_price,
                    "status": order.status,
                    "filled_quantity": order.filled_quantity,
                    "created_at": order.created_at.isoformat(),
                    "updated_at": order.updated_at.isoformat()
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener órdenes abiertas: {e}")
            return []
    
    async def get_closed_orders(self, account_id: int, symbol: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener órdenes cerradas.
        
        Args:
            account_id: ID de la cuenta
            symbol: Símbolo de trading (opcional)
            limit: Número máximo de órdenes a retornar
            
        Returns:
            Lista de órdenes cerradas
        """
        try:
            session = self.Session()
            
            query = session.query(PaperTradingOrder).filter(
                and_(
                    PaperTradingOrder.account_id == account_id,
                    PaperTradingOrder.status.in_(["closed", "canceled"])
                )
            ).order_by(PaperTradingOrder.updated_at.desc()).limit(limit)
            
            if symbol:
                query = query.filter(PaperTradingOrder.symbol == symbol)
            
            orders = query.all()
            
            result = []
            for order in orders:
                result.append({
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.type,
                    "quantity": order.quantity,
                    "price": order.price,
                    "average_price": order.average_price,
                    "status": order.status,
                    "filled_quantity": order.filled_quantity,
                    "created_at": order.created_at.isoformat(),
                    "updated_at": order.updated_at.isoformat()
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener órdenes cerradas: {e}")
            return []
    
    async def get_recent_trades(self, account_id: int, symbol: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener operaciones recientes.
        
        Args:
            account_id: ID de la cuenta
            symbol: Símbolo de trading (opcional)
            limit: Número máximo de operaciones a retornar
            
        Returns:
            Lista de operaciones recientes
        """
        try:
            session = self.Session()
            
            query = session.query(PaperTradingTrade).filter(
                PaperTradingTrade.account_id == account_id
            ).order_by(PaperTradingTrade.timestamp.desc()).limit(limit)
            
            if symbol:
                query = query.filter(PaperTradingTrade.symbol == symbol)
            
            trades = query.all()
            
            result = []
            for trade in trades:
                result.append({
                    "trade_id": trade.trade_id,
                    "order_id": trade.order_id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "commission": trade.commission,
                    "commission_asset": trade.commission_asset,
                    "timestamp": trade.timestamp.isoformat()
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener operaciones recientes: {e}")
            return []
    
    async def calculate_account_value(self, account_id: int) -> float:
        """
        Calcular el valor total de la cuenta en USD.
        
        Args:
            account_id: ID de la cuenta
            
        Returns:
            Valor total de la cuenta en USD
        """
        try:
            # Obtener saldos
            balances = await self.get_account_balance(account_id)
            
            total_usd_value = 0.0
            
            for asset, balance in balances.items():
                total_quantity = balance["total"]
                
                if total_quantity <= 0:
                    continue
                
                # Para USDT, valor es 1:1
                if asset == "USDT":
                    total_usd_value += total_quantity
                else:
                    # Para otros activos, buscar precio en USD
                    symbol = f"{asset}/USDT"
                    price = self.current_prices.get(symbol)
                    
                    if price:
                        total_usd_value += total_quantity * price
                    else:
                        # Buscar en datos históricos recientes
                        session = self.Session()
                        latest_data = session.query(MarketData).filter(
                            MarketData.symbol == symbol
                        ).order_by(desc(MarketData.timestamp)).first()
                        
                        if latest_data:
                            total_usd_value += total_quantity * latest_data.close
                        
                        session.close()
            
            return total_usd_value
            
        except Exception as e:
            self.logger.error(f"Error al calcular valor de cuenta: {e}")
            return 0.0
    
    async def get_historical_prices(self, symbol: str, timeframe: str = '1h', 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener precios históricos para paper trading.
        
        Args:
            symbol: Símbolo de trading
            timeframe: Marco temporal
            limit: Número máximo de registros
            
        Returns:
            Lista de datos OHLCV
        """
        try:
            session = self.Session()
            
            query = session.query(MarketData).filter(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                    MarketData.source == 'testnet'
                )
            ).order_by(desc(MarketData.timestamp)).limit(limit)
            
            data = query.all()
            data.reverse()  # Ordenar cronológicamente
            
            result = []
            for item in data:
                result.append({
                    "timestamp": int(item.timestamp.timestamp() * 1000),
                    "datetime": item.timestamp.isoformat(),
                    "open": item.open,
                    "high": item.high,
                    "low": item.low,
                    "close": item.close,
                    "volume": item.volume
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener precios históricos: {e}")
            return []