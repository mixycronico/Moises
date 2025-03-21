"""
Sistema de Paper Trading para Genesis.

Este módulo proporciona las funcionalidades para ejecutar operaciones simuladas
con datos reales, permitiendo probar estrategias sin riesgo financiero.
"""

import asyncio
import logging
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from genesis.core.base import Component
from genesis.config.settings import settings
from genesis.db.repository import Repository
from genesis.db.paper_trading_models import (
    PaperTradingAccount, PaperAssetBalance, 
    PaperOrder, PaperTrade, PaperBalanceSnapshot
)
from genesis.exchanges.ccxt_wrapper import CCXTExchange

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
        self.logger = logging.getLogger(__name__)
        self.accounts: Dict[int, Dict[str, Any]] = {}  # Cache de cuentas activas
        self.orders: Dict[str, Dict[str, Any]] = {}  # Órdenes abiertas en memoria
        self.data_provider = None  # Proveedor de datos de mercado
        self.repo = Repository()
        self.running = False
        self.update_task = None
        self.snapshot_task = None
        
    async def start(self) -> None:
        """Iniciar el gestor de paper trading."""
        await super().start()
        
        self.logger.info("Iniciando PaperTradingManager")
        
        # Cargar cuentas activas
        await self._load_accounts()
        
        # Iniciar tareas de procesamiento
        self.running = True
        self.update_task = asyncio.create_task(self._process_orders_loop())
        self.snapshot_task = asyncio.create_task(self._daily_snapshot_loop())
        
        self.logger.info("PaperTradingManager iniciado")
        
    async def stop(self) -> None:
        """Detener el gestor de paper trading."""
        self.logger.info("Deteniendo PaperTradingManager")
        
        self.running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            
        if self.snapshot_task:
            self.snapshot_task.cancel()
            try:
                await self.snapshot_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        
        self.logger.info("PaperTradingManager detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente que envió el evento
        """
        if event_type == "trading.order_request" and settings.get("trading.dry_run", True):
            await self._handle_order_request(data)
        elif event_type == "trading.cancel_order_request" and settings.get("trading.dry_run", True):
            await self._handle_cancel_request(data)
        
    async def create_account(self, name: str, initial_balance_usd: float = 10000.0,
                           user_id: Optional[int] = None, description: Optional[str] = None) -> int:
        """
        Crear una nueva cuenta de paper trading.
        
        Args:
            name: Nombre de la cuenta
            initial_balance_usd: Balance inicial en USD
            user_id: ID del usuario propietario (opcional)
            description: Descripción de la cuenta
            
        Returns:
            ID de la cuenta creada
        """
        # Crear cuenta en la base de datos
        account = PaperTradingAccount(
            name=name,
            user_id=user_id,
            description=description,
            initial_balance_usd=initial_balance_usd,
            current_balance_usd=initial_balance_usd,
            config={
                "fee_rate": settings.get("trading.fee_rate", 0.001),
                "slippage": settings.get("trading.slippage", 0.0005),
            }
        )
        
        account_id = await self.repo.create(account)
        
        # Crear balance inicial en USDT
        balance = PaperAssetBalance(
            account_id=account_id,
            asset="USDT",
            total=initial_balance_usd,
            available=initial_balance_usd,
            locked=0.0
        )
        
        await self.repo.create(balance)
        
        # Agregar a cache
        account_data = {
            "id": account_id,
            "name": name,
            "user_id": user_id,
            "initial_balance_usd": initial_balance_usd,
            "current_balance_usd": initial_balance_usd,
            "config": account.config,
            "balances": {"USDT": initial_balance_usd}
        }
        
        self.accounts[account_id] = account_data
        
        self.logger.info(f"Creada cuenta de paper trading: {name} (ID: {account_id})")
        
        return account_id
        
    async def get_account(self, account_id: int) -> Dict[str, Any]:
        """
        Obtener datos de una cuenta de paper trading.
        
        Args:
            account_id: ID de la cuenta
            
        Returns:
            Datos de la cuenta
        """
        # Verificar cache
        if account_id in self.accounts:
            # Actualizar balances desde DB para evitar inconsistencias
            await self._update_account_balances(account_id)
            return self.accounts[account_id]
            
        # Buscar en la base de datos
        account = await self.repo.get_by_id(PaperTradingAccount, account_id)
        if not account:
            raise ValueError(f"Cuenta no encontrada: {account_id}")
            
        # Obtener balances
        balances = await self.repo.query(
            PaperAssetBalance,
            f"account_id = {account_id}"
        )
        
        # Crear datos en cache
        balances_dict = {b.asset: b.total for b in balances}
        
        account_data = {
            "id": account.id,
            "name": account.name,
            "user_id": account.user_id,
            "initial_balance_usd": account.initial_balance_usd,
            "current_balance_usd": account.current_balance_usd,
            "config": account.config,
            "balances": balances_dict
        }
        
        self.accounts[account_id] = account_data
        
        return account_data
        
    async def create_order(self, account_id: int, symbol: str, order_type: str, 
                          side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Crear una orden simulada.
        
        Args:
            account_id: ID de la cuenta
            symbol: Par de trading (ej: 'BTC/USDT')
            order_type: Tipo de orden ('limit', 'market')
            side: Lado de la orden ('buy', 'sell')
            amount: Cantidad a operar
            price: Precio límite (para órdenes limit)
            
        Returns:
            Datos de la orden creada
        """
        # Validaciones básicas
        if side not in ["buy", "sell"]:
            raise ValueError(f"Lado de orden inválido: {side}")
            
        if order_type not in ["limit", "market"]:
            raise ValueError(f"Tipo de orden inválido: {order_type}")
            
        if amount <= 0:
            raise ValueError("La cantidad debe ser mayor que 0")
            
        if order_type == "limit" and (price is None or price <= 0):
            raise ValueError("Las órdenes limit requieren un precio válido")
        
        # Obtener datos de la cuenta
        account_data = await self.get_account(account_id)
        
        # Validar que haya suficiente balance
        await self._validate_balance(account_id, symbol, side, amount, price)
        
        # Generar ID de orden único
        order_id = f"pt_{uuid.uuid4().hex}"
        
        # Crear orden en la base de datos
        base, quote = symbol.split('/')
        
        # Para órdenes market necesitamos el precio actual (para simulación)
        current_price = None
        if order_type == "market" or price is None:
            current_price = await self._get_current_price(symbol)
            if not current_price:
                raise ValueError(f"No se pudo obtener precio actual para {symbol}")
                
            price = current_price
        
        # Crear orden en DB
        order = PaperOrder(
            order_id=order_id,
            account_id=account_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            price=price,
            amount=amount,
            filled=0.0,
            remaining=amount,
            status="open",
            extra={
                "current_price": current_price
            }
        )
        
        await self.repo.create(order)
        
        # Bloquear fondos
        if side == "buy":
            asset = quote
            # Calcular costos incluyendo fees
            fee_rate = account_data["config"].get("fee_rate", 0.001)
            cost = amount * price * (1 + fee_rate)
            
            # Actualizar balance
            await self._update_balance(account_id, asset, 0, -cost)
        else:  # sell
            asset = base
            # Bloquear activo a vender
            await self._update_balance(account_id, asset, 0, -amount)
        
        # Mantener orden en memoria para procesamiento rápido
        order_data = {
            "order_id": order_id,
            "account_id": account_id,
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "price": price,
            "amount": amount,
            "filled": 0.0,
            "remaining": amount,
            "status": "open",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        self.orders[order_id] = order_data
        
        self.logger.info(f"Creada orden paper trading: {order_id} ({side} {amount} {symbol})")
        
        # Procesar inmediatamente si es una orden market
        if order_type == "market":
            await self._execute_order(order_id, current_price)
        
        # Emitir evento
        await self.emit_event("trading.order_created", {
            "order_id": order_id,
            "account_id": account_id,
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "price": price,
            "amount": amount,
            "status": "open"
        })
        
        return order_data
        
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancelar una orden simulada.
        
        Args:
            order_id: ID de la orden a cancelar
            
        Returns:
            Resultado de la cancelación
        """
        # Verificar que la orden exista
        if order_id in self.orders:
            order_data = self.orders[order_id]
        else:
            # Buscar en la base de datos
            order = await self.repo.get_by_field(PaperOrder, "order_id", order_id)
            if not order:
                raise ValueError(f"Orden no encontrada: {order_id}")
                
            if order.status != "open":
                raise ValueError(f"Solo se pueden cancelar órdenes abiertas (estado actual: {order.status})")
                
            # Convertir a dict
            order_data = {
                "order_id": order.order_id,
                "account_id": order.account_id,
                "symbol": order.symbol,
                "order_type": order.order_type,
                "side": order.side,
                "price": order.price,
                "amount": order.amount,
                "filled": order.filled,
                "remaining": order.remaining,
                "status": order.status
            }
        
        # Solo se pueden cancelar órdenes abiertas
        if order_data["status"] != "open":
            raise ValueError(f"Solo se pueden cancelar órdenes abiertas (estado actual: {order_data['status']})")
        
        # Actualizar estado en DB
        await self.repo.execute_query(
            f"UPDATE paper_orders SET status = 'canceled', updated_at = '{datetime.utcnow().isoformat()}', "
            f"closed_at = '{datetime.utcnow().isoformat()}' WHERE order_id = '{order_id}'"
        )
        
        # Desbloquear fondos
        account_id = order_data["account_id"]
        symbol = order_data["symbol"]
        side = order_data["side"]
        remaining = order_data["remaining"]
        price = order_data["price"]
        
        base, quote = symbol.split('/')
        
        if side == "buy":
            asset = quote
            # Calcular costos de la parte no ejecutada incluyendo fees
            account_data = await self.get_account(account_id)
            fee_rate = account_data["config"].get("fee_rate", 0.001)
            cost = remaining * price * (1 + fee_rate)
            
            # Liberar fondos
            await self._update_balance(account_id, asset, cost, 0)
        else:  # sell
            asset = base
            # Liberar activo no vendido
            await self._update_balance(account_id, asset, remaining, 0)
        
        # Actualizar en memoria
        if order_id in self.orders:
            self.orders[order_id]["status"] = "canceled"
            self.orders[order_id]["updated_at"] = datetime.utcnow()
            self.orders[order_id]["closed_at"] = datetime.utcnow()
        
        self.logger.info(f"Cancelada orden paper trading: {order_id}")
        
        # Emitir evento
        await self.emit_event("trading.order_canceled", {
            "order_id": order_id,
            "account_id": account_id,
            "symbol": symbol,
            "status": "canceled"
        })
        
        return {
            "order_id": order_id,
            "status": "canceled",
            "message": "Orden cancelada con éxito"
        }
        
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Obtener datos de una orden simulada.
        
        Args:
            order_id: ID de la orden
            
        Returns:
            Datos de la orden
        """
        # Verificar cache
        if order_id in self.orders:
            return self.orders[order_id]
            
        # Buscar en la base de datos
        order = await self.repo.get_by_field(PaperOrder, "order_id", order_id)
        if not order:
            raise ValueError(f"Orden no encontrada: {order_id}")
            
        # Convertir a dict
        order_data = {
            "order_id": order.order_id,
            "account_id": order.account_id,
            "symbol": order.symbol,
            "order_type": order.order_type,
            "side": order.side,
            "price": order.price,
            "amount": order.amount,
            "filled": order.filled,
            "remaining": order.remaining,
            "status": order.status,
            "created_at": order.created_at,
            "updated_at": order.updated_at,
            "closed_at": order.closed_at
        }
        
        # Agregar al cache si está abierta
        if order.status == "open":
            self.orders[order_id] = order_data
            
        return order_data
        
    async def get_orders(self, account_id: int, symbol: Optional[str] = None, 
                       status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtener órdenes de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            symbol: Filtrar por símbolo (opcional)
            status: Filtrar por estado (opcional)
            
        Returns:
            Lista de órdenes
        """
        query = f"account_id = {account_id}"
        
        if symbol:
            query += f" AND symbol = '{symbol}'"
            
        if status:
            query += f" AND status = '{status}'"
            
        orders = await self.repo.query(PaperOrder, query)
        
        return [{
            "order_id": order.order_id,
            "account_id": order.account_id,
            "symbol": order.symbol,
            "order_type": order.order_type,
            "side": order.side,
            "price": order.price,
            "amount": order.amount,
            "filled": order.filled,
            "remaining": order.remaining,
            "status": order.status,
            "created_at": order.created_at,
            "updated_at": order.updated_at,
            "closed_at": order.closed_at
        } for order in orders]
        
    async def get_trades(self, account_id: int, symbol: Optional[str] = None,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener operaciones ejecutadas de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            symbol: Filtrar por símbolo (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de operaciones
        """
        query = f"account_id = {account_id}"
        
        if symbol:
            query += f" AND symbol = '{symbol}'"
            
        query += f" ORDER BY created_at DESC LIMIT {limit}"
            
        trades = await self.repo.query(PaperTrade, query)
        
        return [{
            "trade_id": trade.trade_id,
            "account_id": trade.account_id,
            "order_id": trade.order_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "price": trade.price,
            "amount": trade.amount,
            "cost": trade.cost,
            "fee": trade.fee,
            "fee_currency": trade.fee_currency,
            "created_at": trade.created_at
        } for trade in trades]
        
    async def get_balance(self, account_id: int, asset: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Obtener el balance de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            asset: Activo específico (opcional)
            
        Returns:
            Diccionario con balances por activo
        """
        query = f"account_id = {account_id}"
        
        if asset:
            query += f" AND asset = '{asset}'"
            
        balances = await self.repo.query(PaperAssetBalance, query)
        
        result = {}
        for balance in balances:
            result[balance.asset] = {
                "total": balance.total,
                "available": balance.available,
                "locked": balance.locked
            }
            
        return result
        
    async def get_performance(self, account_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Obtener rendimiento histórico de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            days: Número de días a considerar
            
        Returns:
            Datos de rendimiento
        """
        # Obtener cuenta
        account = await self.repo.get_by_id(PaperTradingAccount, account_id)
        if not account:
            raise ValueError(f"Cuenta no encontrada: {account_id}")
            
        # Calcular fecha de inicio
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Obtener snapshots en el rango
        query = f"account_id = {account_id} AND timestamp >= '{start_date.isoformat()}' ORDER BY timestamp"
        snapshots = await self.repo.query(PaperBalanceSnapshot, query)
        
        # Calcular métricas
        if not snapshots:
            return {
                "account_id": account_id,
                "initial_balance": account.initial_balance_usd,
                "current_balance": account.current_balance_usd,
                "total_pnl": account.current_balance_usd - account.initial_balance_usd,
                "total_pnl_pct": ((account.current_balance_usd / account.initial_balance_usd) - 1) * 100 if account.initial_balance_usd > 0 else 0,
                "snapshots": []
            }
            
        # Preparar datos para gráfica
        snapshot_data = []
        for snap in snapshots:
            snapshot_data.append({
                "timestamp": snap.timestamp.isoformat(),
                "balance": snap.balance_usd,
                "daily_pnl": snap.daily_pnl,
                "total_pnl": snap.total_pnl,
                "total_pnl_pct": snap.total_pnl_pct
            })
            
        return {
            "account_id": account_id,
            "initial_balance": account.initial_balance_usd,
            "current_balance": account.current_balance_usd,
            "total_pnl": account.current_balance_usd - account.initial_balance_usd,
            "total_pnl_pct": ((account.current_balance_usd / account.initial_balance_usd) - 1) * 100 if account.initial_balance_usd > 0 else 0,
            "snapshots": snapshot_data
        }
        
    async def ensure_default_account(self) -> int:
        """
        Asegurarse de que exista al menos una cuenta para paper trading.
        
        Si no existe ninguna cuenta, crea una cuenta por defecto.
        
        Returns:
            ID de la cuenta por defecto
        """
        # Verificar si ya existe alguna cuenta
        accounts = await self.repo.query(PaperTradingAccount, "is_active = 1")
        
        if accounts:
            # Ya existe al menos una cuenta
            default_account = accounts[0]
            self.logger.info(f"Usando cuenta existente para paper trading: {default_account.name} (ID: {default_account.id})")
            return default_account.id
        
        # No existen cuentas, crear una nueva
        account_id = await self.create_account(
            name="Default Paper Trading Account",
            initial_balance_usd=10000.0,
            description="Cuenta por defecto para paper trading"
        )
        
        self.logger.info(f"Creada nueva cuenta por defecto para paper trading (ID: {account_id})")
        return account_id
    
    async def _load_accounts(self) -> None:
        """Cargar cuentas activas en memoria."""
        accounts = await self.repo.query(PaperTradingAccount, "is_active = 1")
        
        for account in accounts:
            # Obtener balances
            balances = await self.repo.query(
                PaperAssetBalance,
                f"account_id = {account.id}"
            )
            
            # Crear datos en cache
            balances_dict = {b.asset: b.total for b in balances}
            
            account_data = {
                "id": account.id,
                "name": account.name,
                "user_id": account.user_id,
                "initial_balance_usd": account.initial_balance_usd,
                "current_balance_usd": account.current_balance_usd,
                "config": account.config,
                "balances": balances_dict
            }
            
            self.accounts[account.id] = account_data
            
        self.logger.info(f"Cargadas {len(accounts)} cuentas de paper trading")
        
        # Cargar órdenes abiertas
        open_orders = await self.repo.query(PaperOrder, "status = 'open'")
        
        for order in open_orders:
            order_data = {
                "order_id": order.order_id,
                "account_id": order.account_id,
                "symbol": order.symbol,
                "order_type": order.order_type,
                "side": order.side,
                "price": order.price,
                "amount": order.amount,
                "filled": order.filled,
                "remaining": order.remaining,
                "status": order.status,
                "created_at": order.created_at,
                "updated_at": order.updated_at,
                "closed_at": order.closed_at
            }
            
            self.orders[order.order_id] = order_data
            
        self.logger.info(f"Cargadas {len(open_orders)} órdenes abiertas de paper trading")
        
    async def _update_account_balances(self, account_id: int) -> None:
        """
        Actualizar balances de una cuenta desde la base de datos.
        
        Args:
            account_id: ID de la cuenta a actualizar
        """
        if account_id not in self.accounts:
            return
            
        # Obtener balances de la base de datos
        balances = await self.repo.query(
            PaperAssetBalance,
            f"account_id = {account_id}"
        )
        
        # Actualizar cache
        balances_dict = {b.asset: b.total for b in balances}
        self.accounts[account_id]["balances"] = balances_dict
        
        # Actualizar balance total
        account = await self.repo.get_by_id(PaperTradingAccount, account_id)
        if account:
            self.accounts[account_id]["current_balance_usd"] = account.current_balance_usd
        
    async def _validate_balance(self, account_id: int, symbol: str, side: str, 
                            amount: float, price: Optional[float]) -> bool:
        """
        Validar que haya suficiente balance para una operación.
        
        Args:
            account_id: ID de la cuenta
            symbol: Par de trading
            side: Lado de la operación
            amount: Cantidad a operar
            price: Precio límite
            
        Returns:
            True si hay suficiente balance, False en caso contrario
        """
        # Obtener balances actuales
        balances = await self.get_balance(account_id)
        
        base, quote = symbol.split('/')
        
        if side == "buy":
            # Verificar balance en la moneda de cotización (USDT, etc)
            if price is None:
                # Para market orders, necesitamos obtener precio actual
                price = await self._get_current_price(symbol)
                if not price:
                    raise ValueError(f"No se pudo obtener precio actual para {symbol}")
            
            # Calcular costo total incluyendo fees
            account_data = await self.get_account(account_id)
            fee_rate = account_data["config"].get("fee_rate", 0.001)
            cost = amount * price * (1 + fee_rate)
            
            # Verificar balance disponible
            if quote not in balances:
                raise ValueError(f"Balance insuficiente: {quote}")
                
            available = balances[quote]["available"]
            
            if available < cost:
                raise ValueError(f"Balance insuficiente: {available} {quote} < {cost} {quote}")
        
        else:  # sell
            # Verificar balance en la moneda base (BTC, etc)
            if base not in balances:
                raise ValueError(f"Balance insuficiente: {base}")
                
            available = balances[base]["available"]
            
            if available < amount:
                raise ValueError(f"Balance insuficiente: {available} {base} < {amount} {base}")
        
        return True
        
    async def _update_balance(self, account_id: int, asset: str, 
                         available_delta: float, locked_delta: float) -> None:
        """
        Actualizar el balance de un activo en una cuenta.
        
        Args:
            account_id: ID de la cuenta
            asset: Activo a modificar
            available_delta: Cambio en el balance disponible
            locked_delta: Cambio en el balance bloqueado
        """
        # Buscar balance actual
        balance = await self.repo.get_filtered_one(
            PaperAssetBalance, 
            f"account_id = {account_id} AND asset = '{asset}'"
        )
        
        if not balance:
            # Crear nuevo registro si no existe
            balance = PaperAssetBalance(
                account_id=account_id,
                asset=asset,
                total=available_delta,
                available=available_delta,
                locked=locked_delta
            )
            await self.repo.create(balance)
            
            # Actualizar cache
            if account_id in self.accounts:
                if "balances" not in self.accounts[account_id]:
                    self.accounts[account_id]["balances"] = {}
                    
                self.accounts[account_id]["balances"][asset] = available_delta + locked_delta
                
            return
            
        # Actualizar balance existente
        new_available = balance.available + available_delta
        new_locked = balance.locked + locked_delta
        new_total = new_available + new_locked
        
        # Validar que no quede negativo
        if new_available < 0 or new_locked < 0 or new_total < 0:
            self.logger.warning(f"Intento de balance negativo: account_id={account_id}, asset={asset}, "
                              f"available_delta={available_delta}, locked_delta={locked_delta}")
            raise ValueError(f"El balance no puede ser negativo: {asset}")
            
        # Actualizar en DB
        query = (
            f"UPDATE paper_asset_balances SET "
            f"available = {new_available}, "
            f"locked = {new_locked}, "
            f"total = {new_total}, "
            f"updated_at = '{datetime.utcnow().isoformat()}' "
            f"WHERE account_id = {account_id} AND asset = '{asset}'"
        )
        
        await self.repo.execute_query(query)
        
        # Actualizar cache
        if account_id in self.accounts:
            if "balances" not in self.accounts[account_id]:
                self.accounts[account_id]["balances"] = {}
                
            self.accounts[account_id]["balances"][asset] = new_total
        
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Obtener el precio actual de un símbolo.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Precio actual o None si no está disponible
        """
        # Si tenemos acceso directo a exchange
        try:
            # Intentar usar el data provider si está disponible
            if self.data_provider and hasattr(self.data_provider, "get_ticker"):
                ticker = await self.data_provider.get_ticker(symbol)
                if ticker and "last" in ticker:
                    return ticker["last"]
                    
            # Crear exchange temporal
            exchange = CCXTExchange(
                exchange_id="binance",
                config={"testnet": True}
            )
            
            await exchange.start()
            
            try:
                ticker = await exchange.fetch_ticker(symbol)
                return ticker["last"]
            finally:
                await exchange.stop()
                
        except Exception as e:
            self.logger.error(f"Error al obtener precio actual para {symbol}: {e}")
            
        return None
        
    async def _execute_order(self, order_id: str, current_price: Optional[float] = None) -> bool:
        """
        Ejecutar una orden simulada.
        
        Args:
            order_id: ID de la orden
            current_price: Precio actual (opcional)
            
        Returns:
            True si la orden se ejecutó, False en caso contrario
        """
        # Obtener datos de la orden
        if order_id in self.orders:
            order_data = self.orders[order_id]
        else:
            order = await self.repo.get_by_field(PaperOrder, "order_id", order_id)
            if not order or order.status != "open":
                return False
                
            order_data = {
                "order_id": order.order_id,
                "account_id": order.account_id,
                "symbol": order.symbol,
                "order_type": order.order_type,
                "side": order.side,
                "price": order.price,
                "amount": order.amount,
                "filled": order.filled,
                "remaining": order.remaining,
                "status": order.status,
                "created_at": order.created_at
            }
        
        # Si la orden no está abierta, no hacer nada
        if order_data["status"] != "open":
            return False
            
        # Obtener precio actual si no se proporcionó
        if not current_price:
            current_price = await self._get_current_price(order_data["symbol"])
            if not current_price:
                self.logger.warning(f"No se pudo obtener precio actual para {order_data['symbol']}")
                return False
                
        # Verificar si la orden se debe ejecutar
        execute = False
        execution_price = None
        
        if order_data["order_type"] == "market":
            # Las órdenes market se ejecutan inmediatamente al precio actual
            execute = True
            execution_price = current_price
            
        elif order_data["order_type"] == "limit":
            # Las órdenes limit se ejecutan si el precio actual es favorable
            if order_data["side"] == "buy" and current_price <= order_data["price"]:
                execute = True
                execution_price = order_data["price"]
            elif order_data["side"] == "sell" and current_price >= order_data["price"]:
                execute = True
                execution_price = order_data["price"]
                
        if not execute:
            return False
            
        # Ejecutar la orden
        account_id = order_data["account_id"]
        symbol = order_data["symbol"]
        side = order_data["side"]
        amount = order_data["remaining"]
        
        # Aplicar slippage para simular condiciones reales
        account_data = await self.get_account(account_id)
        slippage_factor = account_data["config"].get("slippage", 0.0005)  # 0.05% por defecto
        
        if side == "buy":
            # En compras, el slippage incrementa el precio (es decir, compramos un poco más caro)
            execution_price = execution_price * (1 + slippage_factor)
        else:
            # En ventas, el slippage reduce el precio (es decir, vendemos un poco más barato)
            execution_price = execution_price * (1 - slippage_factor)
        
        # Calcular fee
        fee_rate = account_data["config"].get("fee_rate", 0.001)
        fee_amount = amount * execution_price * fee_rate
        fee_currency = symbol.split('/')[1]  # La fee suele ser en la moneda quote
        
        # Generar ID único para el trade
        trade_id = f"pt_trade_{uuid.uuid4().hex}"
        
        # Registrar el trade
        trade = PaperTrade(
            trade_id=trade_id,
            account_id=account_id,
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=execution_price,
            amount=amount,
            cost=amount * execution_price,
            fee=fee_amount,
            fee_currency=fee_currency,
            strategy_id=None  # Por ahora no hay estrategia asociada
        )
        
        await self.repo.create(trade)
        
        # Actualizar orden
        await self.repo.execute_query(
            f"UPDATE paper_orders SET "
            f"filled = amount, "
            f"remaining = 0, "
            f"status = 'closed', "
            f"updated_at = '{datetime.utcnow().isoformat()}', "
            f"closed_at = '{datetime.utcnow().isoformat()}' "
            f"WHERE order_id = '{order_id}'"
        )
        
        # Actualizar balances
        base, quote = symbol.split('/')
        
        if side == "buy":
            # Transferir la moneda base a la cuenta
            await self._update_balance(account_id, base, amount, 0)
            
            # Los fondos de quote ya fueron bloqueados al crear la orden
            # Ajustar la diferencia por el cambio de precio y fee si es necesario
        else:  # sell
            # Desbloquear la moneda base (ya había sido bloqueada)
            # No es necesario hacer nada, ya que al crear la orden se tomó del available
            
            # Añadir la moneda quote recibida por la venta
            sale_amount = amount * execution_price - fee_amount
            await self._update_balance(account_id, quote, sale_amount, 0)
        
        # Actualizar balance total de la cuenta
        await self._update_account_total_balance(account_id)
        
        # Actualizar en memoria
        if order_id in self.orders:
            self.orders[order_id]["filled"] = order_data["amount"]
            self.orders[order_id]["remaining"] = 0
            self.orders[order_id]["status"] = "closed"
            self.orders[order_id]["updated_at"] = datetime.utcnow()
            self.orders[order_id]["closed_at"] = datetime.utcnow()
        
        self.logger.info(f"Ejecutada orden paper trading: {order_id} ({side} {amount} {symbol} @ {execution_price})")
        
        # Emitir evento
        await self.emit_event("trading.order_filled", {
            "order_id": order_id,
            "trade_id": trade_id,
            "account_id": account_id,
            "symbol": symbol,
            "side": side,
            "price": execution_price,
            "amount": amount,
            "fee": fee_amount,
            "fee_currency": fee_currency,
            "status": "closed"
        })
        
        return True
        
    async def _update_account_total_balance(self, account_id: int) -> float:
        """
        Actualizar el balance total de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            
        Returns:
            Nuevo balance total
        """
        # Obtener todos los balances
        balances = await self.get_balance(account_id)
        
        # Calcular valor total en USD
        total_usd = 0.0
        
        for asset, data in balances.items():
            asset_total = data["total"]
            
            if asset == "USDT":
                # USDT se considera 1:1 con USD
                total_usd += asset_total
            else:
                # Para otros activos, necesitamos obtener el precio actual
                try:
                    symbol = f"{asset}/USDT"
                    price = await self._get_current_price(symbol)
                    if price:
                        total_usd += asset_total * price
                except Exception as e:
                    self.logger.warning(f"No se pudo obtener precio para {asset}: {e}")
        
        # Actualizar en DB
        await self.repo.execute_query(
            f"UPDATE paper_trading_accounts SET "
            f"current_balance_usd = {total_usd}, "
            f"updated_at = '{datetime.utcnow().isoformat()}' "
            f"WHERE id = {account_id}"
        )
        
        # Actualizar en cache
        if account_id in self.accounts:
            self.accounts[account_id]["current_balance_usd"] = total_usd
            
        return total_usd
        
    async def _process_orders_loop(self) -> None:
        """Procesar órdenes abiertas periódicamente."""
        self.logger.info("Iniciando procesamiento periódico de órdenes")
        
        while self.running:
            try:
                # Obtener todas las órdenes abiertas
                open_orders = list(self.orders.values())
                
                # Agrupar por símbolo para reducir consultas de precio
                orders_by_symbol = {}
                for order in open_orders:
                    if order["status"] != "open":
                        continue
                        
                    symbol = order["symbol"]
                    if symbol not in orders_by_symbol:
                        orders_by_symbol[symbol] = []
                        
                    orders_by_symbol[symbol].append(order)
                
                # Procesar órdenes por símbolo
                for symbol, symbol_orders in orders_by_symbol.items():
                    if not symbol_orders:
                        continue
                        
                    # Obtener precio actual
                    current_price = await self._get_current_price(symbol)
                    if not current_price:
                        continue
                        
                    # Procesar órdenes con este precio
                    for order in symbol_orders:
                        await self._execute_order(order["order_id"], current_price)
                        
                # Esperar antes de la siguiente iteración
                await asyncio.sleep(5)  # Procesar cada 5 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error en procesamiento de órdenes: {e}")
                await asyncio.sleep(5)  # En caso de error, esperar antes de reintentar
                
        self.logger.info("Detenido procesamiento periódico de órdenes")
        
    async def _daily_snapshot_loop(self) -> None:
        """Crear instantáneas diarias de balances."""
        self.logger.info("Iniciando creación de instantáneas diarias")
        
        # Calcular próxima ejecución (medianoche)
        now = datetime.utcnow()
        next_run = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        # Esperar hasta medianoche
        initial_delay = (next_run - now).total_seconds()
        await asyncio.sleep(initial_delay)
        
        while self.running:
            try:
                # Obtener todas las cuentas activas
                accounts = await self.repo.query(PaperTradingAccount, "is_active = 1")
                
                # Crear snapshot para cada cuenta
                for account in accounts:
                    try:
                        # Obtener balances actuales
                        balances = await self.get_balance(account.id)
                        
                        # Obtener snapshot anterior para calcular PnL diario
                        yesterday = datetime.utcnow() - timedelta(days=1)
                        yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
                        
                        prev_snapshot = await self.repo.get_filtered_one(
                            PaperBalanceSnapshot,
                            f"account_id = {account.id} AND timestamp >= '{yesterday.isoformat()}' "
                            f"ORDER BY timestamp DESC LIMIT 1"
                        )
                        
                        daily_pnl = None
                        if prev_snapshot:
                            daily_pnl = account.current_balance_usd - prev_snapshot.balance_usd
                            
                        # Calcular PnL total
                        total_pnl = account.current_balance_usd - account.initial_balance_usd
                        total_pnl_pct = ((account.current_balance_usd / account.initial_balance_usd) - 1) * 100 if account.initial_balance_usd > 0 else 0
                        
                        # Crear snapshot
                        snapshot = PaperBalanceSnapshot(
                            account_id=account.id,
                            timestamp=datetime.utcnow(),
                            balance_usd=account.current_balance_usd,
                            assets={asset: data["total"] for asset, data in balances.items()},
                            daily_pnl=daily_pnl,
                            total_pnl=total_pnl,
                            total_pnl_pct=total_pnl_pct
                        )
                        
                        await self.repo.create(snapshot)
                        
                    except Exception as e:
                        self.logger.error(f"Error creando snapshot para cuenta {account.id}: {e}")
                
                # Esperar hasta el próximo día
                await asyncio.sleep(24 * 60 * 60)  # 24 horas
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error en creación de instantáneas: {e}")
                await asyncio.sleep(60 * 60)  # En caso de error, esperar 1 hora antes de reintentar
                
        self.logger.info("Detenida creación de instantáneas diarias")
        
    async def _handle_order_request(self, data: Dict[str, Any]) -> None:
        """
        Manejar solicitudes de órdenes del sistema.
        
        Args:
            data: Datos de la solicitud
        """
        try:
            # Extraer parámetros
            account_id = data.get("account_id")
            if not account_id:
                # Usar cuenta predeterminada si existe
                accounts = await self.repo.query(PaperTradingAccount, "is_active = 1 LIMIT 1")
                if accounts:
                    account_id = accounts[0].id
                else:
                    # Crear una cuenta por defecto
                    account_id = await self.create_account("Default Paper Account")
            
            symbol = data.get("symbol")
            order_type = data.get("order_type", "market")
            side = data.get("side")
            amount = data.get("amount")
            price = data.get("price")
            
            if not all([symbol, side, amount]):
                self.logger.error(f"Parámetros insuficientes para orden: {data}")
                await self.emit_event("trading.order_error", {
                    "error": "Parámetros insuficientes",
                    "data": data
                })
                return
                
            # Crear orden
            result = await self.create_order(
                account_id=account_id,
                symbol=symbol,
                order_type=order_type,
                side=side,
                amount=amount,
                price=price
            )
            
            # Emitir resultado (adicional al evento emitido en create_order)
            await self.emit_event("trading.order_response", {
                "request": data,
                "result": result,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Error procesando solicitud de orden: {e}")
            
            await self.emit_event("trading.order_error", {
                "error": str(e),
                "data": data
            })
            
    async def _handle_cancel_request(self, data: Dict[str, Any]) -> None:
        """
        Manejar solicitudes de cancelación de órdenes.
        
        Args:
            data: Datos de la solicitud
        """
        try:
            # Extraer parámetros
            order_id = data.get("order_id")
            
            if not order_id:
                self.logger.error(f"Parámetros insuficientes para cancelar orden: {data}")
                await self.emit_event("trading.cancel_error", {
                    "error": "Parámetros insuficientes",
                    "data": data
                })
                return
                
            # Cancelar orden
            result = await self.cancel_order(order_id)
            
            # Emitir resultado (adicional al evento emitido en cancel_order)
            await self.emit_event("trading.cancel_response", {
                "request": data,
                "result": result,
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Error procesando solicitud de cancelación: {e}")
            
            await self.emit_event("trading.cancel_error", {
                "error": str(e),
                "data": data
            })