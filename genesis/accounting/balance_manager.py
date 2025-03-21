"""
Gestor de saldos para el sistema Genesis.

Este módulo proporciona funcionalidades para gestionar los saldos de los usuarios,
incluyendo seguimiento de operaciones, cálculo de rentabilidad, y generación
de informes financieros.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal, getcontext

from genesis.core.base import Component
from genesis.utils.helpers import generate_id, format_timestamp

# Configurar precisión para operaciones con Decimal
getcontext().prec = 28

class AccountTransaction:
    """
    Transacción en una cuenta.
    
    Representa un movimiento de fondos en una cuenta, como un depósito,
    retiro, o resultado de una operación de trading.
    """
    
    def __init__(
        self,
        transaction_id: str,
        account_id: str,
        timestamp: datetime,
        transaction_type: str,
        amount: Decimal,
        currency: str,
        description: str = "",
        related_trade_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar una transacción.
        
        Args:
            transaction_id: Identificador único
            account_id: ID de la cuenta
            timestamp: Fecha y hora de la transacción
            transaction_type: Tipo (deposit, withdrawal, trade_profit, trade_loss, fee, etc.)
            amount: Cantidad (positiva para entradas, negativa para salidas)
            currency: Moneda
            description: Descripción opcional
            related_trade_id: ID de la operación relacionada (si aplica)
            metadata: Metadatos adicionales
        """
        self.transaction_id = transaction_id
        self.account_id = account_id
        self.timestamp = timestamp
        self.transaction_type = transaction_type
        self.amount = Decimal(str(amount))
        self.currency = currency
        self.description = description
        self.related_trade_id = related_trade_id
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos de la transacción
        """
        return {
            "transaction_id": self.transaction_id,
            "account_id": self.account_id,
            "timestamp": format_timestamp(self.timestamp),
            "transaction_type": self.transaction_type,
            "amount": str(self.amount),
            "currency": self.currency,
            "description": self.description,
            "related_trade_id": self.related_trade_id,
            "metadata": self.metadata
        }
        
class Account:
    """
    Cuenta de un usuario o estrategia.
    
    Representa una cuenta en el sistema, manteniendo el saldo, historial
    de transacciones, y métricas de rendimiento.
    """
    
    def __init__(
        self,
        account_id: str,
        name: str,
        owner_id: str,
        initial_balance: Dict[str, Union[float, Decimal]] = None,
        account_type: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar una cuenta.
        
        Args:
            account_id: Identificador único
            name: Nombre de la cuenta
            owner_id: ID del propietario
            initial_balance: Saldo inicial por moneda
            account_type: Tipo de cuenta (user, strategy, system)
            metadata: Metadatos adicionales
        """
        self.account_id = account_id
        self.name = name
        self.owner_id = owner_id
        self.account_type = account_type
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        
        # Inicializar balances
        self.balances = {}
        if initial_balance:
            for currency, amount in initial_balance.items():
                self.balances[currency] = Decimal(str(amount))
                
        # Historial de transacciones y métricas
        self.transactions = []
        self.performance_metrics = {
            "total_deposits": Decimal("0"),
            "total_withdrawals": Decimal("0"),
            "total_profit": Decimal("0"),
            "total_loss": Decimal("0"),
            "total_fees": Decimal("0"),
            "roi": Decimal("0")
        }
        
    def add_transaction(self, transaction: AccountTransaction) -> bool:
        """
        Añadir una transacción a la cuenta.
        
        Args:
            transaction: Transacción a añadir
            
        Returns:
            True si se añadió correctamente, False en caso contrario
        """
        # Verificar que la transacción pertenece a esta cuenta
        if transaction.account_id != self.account_id:
            return False
            
        # Actualizar balance
        currency = transaction.currency
        if currency not in self.balances:
            self.balances[currency] = Decimal("0")
            
        self.balances[currency] += transaction.amount
        
        # Actualizar métricas según el tipo de transacción
        if transaction.transaction_type == "deposit":
            self.performance_metrics["total_deposits"] += transaction.amount
        elif transaction.transaction_type == "withdrawal":
            self.performance_metrics["total_withdrawals"] += abs(transaction.amount)
        elif transaction.transaction_type == "trade_profit":
            self.performance_metrics["total_profit"] += transaction.amount
        elif transaction.transaction_type == "trade_loss":
            self.performance_metrics["total_loss"] += abs(transaction.amount)
        elif transaction.transaction_type == "fee":
            self.performance_metrics["total_fees"] += abs(transaction.amount)
            
        # Calcular ROI
        total_in = self.performance_metrics["total_deposits"]
        if total_in > 0:
            net_profit = (self.performance_metrics["total_profit"] - 
                          self.performance_metrics["total_loss"] - 
                          self.performance_metrics["total_fees"])
            self.performance_metrics["roi"] = (net_profit / total_in) * 100
            
        # Añadir a historial y actualizar timestamp
        self.transactions.append(transaction)
        self.last_updated = datetime.now()
        
        return True
        
    def get_balance(self, currency: str) -> Decimal:
        """
        Obtener el saldo de una moneda.
        
        Args:
            currency: Código de la moneda
            
        Returns:
            Saldo actual
        """
        return self.balances.get(currency, Decimal("0"))
        
    def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_type: Optional[str] = None,
        currency: Optional[str] = None,
        limit: int = 100
    ) -> List[AccountTransaction]:
        """
        Obtener transacciones filtradas.
        
        Args:
            start_date: Fecha inicial
            end_date: Fecha final
            transaction_type: Tipo de transacción
            currency: Moneda
            limit: Límite de resultados
            
        Returns:
            Lista de transacciones
        """
        filtered = self.transactions
        
        if start_date:
            filtered = [t for t in filtered if t.timestamp >= start_date]
            
        if end_date:
            filtered = [t for t in filtered if t.timestamp <= end_date]
            
        if transaction_type:
            filtered = [t for t in filtered if t.transaction_type == transaction_type]
            
        if currency:
            filtered = [t for t in filtered if t.currency == currency]
            
        # Ordenar por fecha descendente y limitar
        filtered.sort(key=lambda t: t.timestamp, reverse=True)
        return filtered[:limit]
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos de la cuenta
        """
        return {
            "account_id": self.account_id,
            "name": self.name,
            "owner_id": self.owner_id,
            "account_type": self.account_type,
            "balances": {k: str(v) for k, v in self.balances.items()},
            "created_at": format_timestamp(self.created_at),
            "last_updated": format_timestamp(self.last_updated),
            "performance_metrics": {k: str(v) for k, v in self.performance_metrics.items()},
            "metadata": self.metadata
        }
        
class BalanceManager(Component):
    """
    Gestor de balances y cuentas.
    
    Este componente gestiona las cuentas de usuarios y estrategias,
    procesa transacciones, y calcula métricas de rendimiento.
    """
    
    def __init__(self, name: str = "balance_manager"):
        """
        Inicializar el gestor de balances.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self.accounts = {}  # account_id -> Account
        self.trade_balances = {}  # trade_id -> Dict[str, Any]
        
    async def start(self) -> None:
        """Iniciar el gestor de balances."""
        await super().start()
        self.logger.info("Gestor de balances iniciado")
        
    async def stop(self) -> None:
        """Detener el gestor de balances."""
        await super().stop()
        self.logger.info("Gestor de balances detenido")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente origen
        """
        if event_type == "trade.opened":
            # Registrar saldo de la operación
            await self._handle_trade_opened(data)
            
        elif event_type == "trade.closed":
            # Actualizar saldo con el resultado de la operación
            await self._handle_trade_closed(data)
            
        elif event_type == "account.deposit":
            # Procesar depósito
            account_id = data.get("account_id")
            amount = data.get("amount")
            currency = data.get("currency")
            
            if account_id and amount is not None and currency:
                await self.deposit(account_id, amount, currency, data.get("description", ""), data.get("metadata"))
                
        elif event_type == "account.withdrawal":
            # Procesar retiro
            account_id = data.get("account_id")
            amount = data.get("amount")
            currency = data.get("currency")
            
            if account_id and amount is not None and currency:
                await self.withdraw(account_id, amount, currency, data.get("description", ""), data.get("metadata"))
                
    async def _handle_trade_opened(self, data: Dict[str, Any]) -> None:
        """
        Manejar evento de operación abierta.
        
        Args:
            data: Datos de la operación
        """
        trade_id = data.get("trade_id")
        account_id = data.get("account_id")
        
        if not trade_id or not account_id:
            self.logger.error("ID de operación o cuenta faltante en evento trade.opened")
            return
            
        # Registrar operación
        self.trade_balances[trade_id] = {
            "trade_id": trade_id,
            "account_id": account_id,
            "entry_price": data.get("entry_price"),
            "amount": data.get("amount"),
            "symbol": data.get("symbol"),
            "side": data.get("side"),
            "timestamp": datetime.now(),
            "entry_fee": data.get("fee", 0)
        }
        
        # Aplicar comisión de entrada si existe
        if data.get("fee"):
            fee = Decimal(str(data.get("fee")))
            currency = data.get("fee_currency", "USDT")
            
            # Crear transacción para la comisión
            transaction = AccountTransaction(
                transaction_id=generate_id(),
                account_id=account_id,
                timestamp=datetime.now(),
                transaction_type="fee",
                amount=-fee,
                currency=currency,
                description=f"Comisión de entrada para operación {trade_id}",
                related_trade_id=trade_id,
                metadata={"fee_type": "entry", "trade_symbol": data.get("symbol")}
            )
            
            # Obtener la cuenta
            account = self.accounts.get(account_id)
            if account:
                account.add_transaction(transaction)
                
                # Emitir evento de actualización
                await self.emit_event("account.updated", {
                    "account_id": account_id,
                    "balance_change": {
                        "currency": currency,
                        "amount": -fee,
                        "reason": "trade_fee"
                    }
                })
        
    async def _handle_trade_closed(self, data: Dict[str, Any]) -> None:
        """
        Manejar evento de operación cerrada.
        
        Args:
            data: Datos de la operación
        """
        trade_id = data.get("trade_id")
        account_id = data.get("account_id")
        
        if not trade_id or not account_id:
            self.logger.error("ID de operación o cuenta faltante en evento trade.closed")
            return
            
        # Obtener detalles de la operación
        trade_entry = self.trade_balances.get(trade_id)
        if not trade_entry:
            self.logger.error(f"No se encontró registro de la operación {trade_id}")
            return
            
        # Calcular rentabilidad
        exit_price = data.get("exit_price")
        amount = Decimal(str(trade_entry.get("amount", 0)))
        entry_price = Decimal(str(trade_entry.get("entry_price", 0)))
        side = trade_entry.get("side", "buy")
        
        if exit_price and amount > 0 and entry_price > 0:
            exit_price = Decimal(str(exit_price))
            
            # Diferentes cálculos según el lado (compra/venta)
            if side == "buy":  # Compra (long)
                pnl = (exit_price - entry_price) * amount
            else:  # Venta (short)
                pnl = (entry_price - exit_price) * amount
                
            # Aplicar comisión de salida si existe
            exit_fee = Decimal(str(data.get("fee", 0)))
            pnl -= exit_fee
            
            # Registro de comisión de salida
            if exit_fee > 0:
                fee_currency = data.get("fee_currency", "USDT")
                
                # Crear transacción para la comisión
                fee_transaction = AccountTransaction(
                    transaction_id=generate_id(),
                    account_id=account_id,
                    timestamp=datetime.now(),
                    transaction_type="fee",
                    amount=-exit_fee,
                    currency=fee_currency,
                    description=f"Comisión de salida para operación {trade_id}",
                    related_trade_id=trade_id,
                    metadata={"fee_type": "exit", "trade_symbol": trade_entry.get("symbol")}
                )
                
                # Añadir transacción a la cuenta
                account = self.accounts.get(account_id)
                if account:
                    account.add_transaction(fee_transaction)
            
            # Crear transacción para el resultado de la operación
            result_type = "trade_profit" if pnl > 0 else "trade_loss"
            result_currency = data.get("profit_currency", "USDT")
            
            result_transaction = AccountTransaction(
                transaction_id=generate_id(),
                account_id=account_id,
                timestamp=datetime.now(),
                transaction_type=result_type,
                amount=pnl,
                currency=result_currency,
                description=f"Resultado de operación {trade_id} - {trade_entry.get('symbol')}",
                related_trade_id=trade_id,
                metadata={
                    "symbol": trade_entry.get("symbol"),
                    "side": side,
                    "entry_price": str(entry_price),
                    "exit_price": str(exit_price),
                    "amount": str(amount),
                    "pnl_percentage": str((pnl / (entry_price * amount)) * 100)
                }
            )
            
            # Añadir transacción a la cuenta
            account = self.accounts.get(account_id)
            if account:
                account.add_transaction(result_transaction)
                
                # Emitir evento de actualización
                await self.emit_event("account.updated", {
                    "account_id": account_id,
                    "balance_change": {
                        "currency": result_currency,
                        "amount": pnl,
                        "reason": "trade_result"
                    },
                    "trade_result": {
                        "trade_id": trade_id,
                        "symbol": trade_entry.get("symbol"),
                        "side": side,
                        "pnl": str(pnl),
                        "pnl_percentage": str((pnl / (entry_price * amount)) * 100)
                    }
                })
                
        # Limpiar entrada de trade_balances
        self.trade_balances.pop(trade_id, None)
        
    async def create_account(
        self,
        name: str,
        owner_id: str,
        initial_balance: Optional[Dict[str, Union[float, Decimal]]] = None,
        account_type: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crear una nueva cuenta.
        
        Args:
            name: Nombre de la cuenta
            owner_id: ID del propietario
            initial_balance: Saldo inicial por moneda
            account_type: Tipo de cuenta
            metadata: Metadatos adicionales
            
        Returns:
            ID de la cuenta creada
        """
        account_id = generate_id()
        
        account = Account(
            account_id=account_id,
            name=name,
            owner_id=owner_id,
            initial_balance=initial_balance,
            account_type=account_type,
            metadata=metadata
        )
        
        self.accounts[account_id] = account
        
        # Registrar depósitos iniciales
        if initial_balance:
            for currency, amount in initial_balance.items():
                decimal_amount = Decimal(str(amount))
                
                # Solo crear transacción si el monto es positivo
                if decimal_amount > 0:
                    transaction = AccountTransaction(
                        transaction_id=generate_id(),
                        account_id=account_id,
                        timestamp=datetime.now(),
                        transaction_type="deposit",
                        amount=decimal_amount,
                        currency=currency,
                        description="Depósito inicial"
                    )
                    
                    account.add_transaction(transaction)
        
        # Emitir evento de cuenta creada
        await self.emit_event("account.created", {
            "account_id": account_id,
            "name": name,
            "owner_id": owner_id,
            "account_type": account_type,
            "initial_balance": {k: str(v) for k, v in (initial_balance or {}).items()}
        })
        
        self.logger.info(f"Cuenta creada: {account_id} para {owner_id}")
        return account_id
        
    async def deposit(
        self,
        account_id: str,
        amount: Union[float, Decimal],
        currency: str,
        description: str = "Depósito",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Realizar un depósito en una cuenta.
        
        Args:
            account_id: ID de la cuenta
            amount: Cantidad a depositar
            currency: Moneda
            description: Descripción
            metadata: Metadatos adicionales
            
        Returns:
            True si se realizó correctamente, False en caso contrario
        """
        if account_id not in self.accounts:
            self.logger.error(f"Cuenta no encontrada: {account_id}")
            return False
            
        decimal_amount = Decimal(str(amount))
        if decimal_amount <= 0:
            self.logger.error(f"Monto de depósito inválido: {amount}")
            return False
            
        # Crear transacción
        transaction = AccountTransaction(
            transaction_id=generate_id(),
            account_id=account_id,
            timestamp=datetime.now(),
            transaction_type="deposit",
            amount=decimal_amount,
            currency=currency,
            description=description,
            metadata=metadata
        )
        
        # Añadir a la cuenta
        result = self.accounts[account_id].add_transaction(transaction)
        
        if result:
            # Emitir evento
            await self.emit_event("account.deposited", {
                "account_id": account_id,
                "amount": str(decimal_amount),
                "currency": currency,
                "transaction_id": transaction.transaction_id,
                "new_balance": str(self.accounts[account_id].get_balance(currency))
            })
            
        return result
        
    async def withdraw(
        self,
        account_id: str,
        amount: Union[float, Decimal],
        currency: str,
        description: str = "Retiro",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Realizar un retiro de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            amount: Cantidad a retirar
            currency: Moneda
            description: Descripción
            metadata: Metadatos adicionales
            
        Returns:
            True si se realizó correctamente, False en caso contrario
        """
        if account_id not in self.accounts:
            self.logger.error(f"Cuenta no encontrada: {account_id}")
            return False
            
        decimal_amount = Decimal(str(amount))
        if decimal_amount <= 0:
            self.logger.error(f"Monto de retiro inválido: {amount}")
            return False
            
        # Verificar saldo suficiente
        account = self.accounts[account_id]
        if account.get_balance(currency) < decimal_amount:
            self.logger.error(f"Saldo insuficiente para retiro: {account.get_balance(currency)} < {decimal_amount}")
            return False
            
        # Crear transacción (monto negativo para retiros)
        transaction = AccountTransaction(
            transaction_id=generate_id(),
            account_id=account_id,
            timestamp=datetime.now(),
            transaction_type="withdrawal",
            amount=-decimal_amount,  # Negativo para retiros
            currency=currency,
            description=description,
            metadata=metadata
        )
        
        # Añadir a la cuenta
        result = account.add_transaction(transaction)
        
        if result:
            # Emitir evento
            await self.emit_event("account.withdrawn", {
                "account_id": account_id,
                "amount": str(decimal_amount),
                "currency": currency,
                "transaction_id": transaction.transaction_id,
                "new_balance": str(account.get_balance(currency))
            })
            
        return result
        
    def get_account(self, account_id: str) -> Optional[Account]:
        """
        Obtener una cuenta por su ID.
        
        Args:
            account_id: ID de la cuenta
            
        Returns:
            Cuenta o None si no existe
        """
        return self.accounts.get(account_id)
        
    def get_accounts_by_owner(self, owner_id: str) -> List[Account]:
        """
        Obtener cuentas por propietario.
        
        Args:
            owner_id: ID del propietario
            
        Returns:
            Lista de cuentas
        """
        return [a for a in self.accounts.values() if a.owner_id == owner_id]
        
    def get_account_balance(self, account_id: str, currency: str) -> Decimal:
        """
        Obtener saldo de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            currency: Moneda
            
        Returns:
            Saldo actual (0 si no existe)
        """
        account = self.get_account(account_id)
        if account:
            return account.get_balance(currency)
        return Decimal("0")
        
    def get_account_transactions(
        self,
        account_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_type: Optional[str] = None,
        currency: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener transacciones de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            start_date: Fecha inicial
            end_date: Fecha final
            transaction_type: Tipo de transacción
            currency: Moneda
            limit: Límite de resultados
            
        Returns:
            Lista de transacciones como diccionarios
        """
        account = self.get_account(account_id)
        if not account:
            return []
            
        transactions = account.get_transactions(
            start_date=start_date,
            end_date=end_date,
            transaction_type=transaction_type,
            currency=currency,
            limit=limit
        )
        
        return [t.to_dict() for t in transactions]
        
    def get_account_performance(self, account_id: str) -> Dict[str, Any]:
        """
        Obtener métricas de rendimiento de una cuenta.
        
        Args:
            account_id: ID de la cuenta
            
        Returns:
            Métricas de rendimiento
        """
        account = self.get_account(account_id)
        if not account:
            return {}
            
        metrics = {k: str(v) for k, v in account.performance_metrics.items()}
        
        # Añadir métricas adicionales
        trades = [t for t in account.transactions if t.transaction_type in ["trade_profit", "trade_loss"]]
        winning_trades = [t for t in trades if t.transaction_type == "trade_profit"]
        losing_trades = [t for t in trades if t.transaction_type == "trade_loss"]
        
        metrics["total_trades"] = str(len(trades))
        metrics["winning_trades"] = str(len(winning_trades))
        metrics["losing_trades"] = str(len(losing_trades))
        
        if trades:
            win_rate = (len(winning_trades) / len(trades)) * 100
            metrics["win_rate"] = str(win_rate)
            
        # Calcular factor de beneficio (profit factor)
        if account.performance_metrics["total_loss"] > 0:
            profit_factor = account.performance_metrics["total_profit"] / account.performance_metrics["total_loss"]
            metrics["profit_factor"] = str(profit_factor)
            
        return metrics