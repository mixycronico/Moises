"""
Gestor de balances para el sistema Genesis.

Este módulo proporciona herramientas para monitorear y gestionar los balances
de diferentes exchanges, wallets o cuentas, llevando un registro detallado de
fondos, transacciones y distribución de activos.
"""

import os
import json
import time
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
import logging
import uuid

from genesis.core.base import Component
from genesis.utils.logger import setup_logging
from genesis.utils.helpers import safe_divide, load_json_file, save_json_file


class Balance:
    """Clase para representar el balance de un activo."""
    
    def __init__(
        self,
        asset: str,
        free: float = 0.0,
        locked: float = 0.0,
        total: Optional[float] = None
    ):
        """
        Inicializar balance.
        
        Args:
            asset: Símbolo del activo
            free: Cantidad disponible
            locked: Cantidad bloqueada
            total: Cantidad total (si no se especifica, se calcula como free + locked)
        """
        self.asset = asset
        self.free = free
        self.locked = locked
        self.total = total if total is not None else free + locked
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos del balance
        """
        return {
            "asset": self.asset,
            "free": self.free,
            "locked": self.locked,
            "total": self.total
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Balance':
        """
        Crear desde diccionario.
        
        Args:
            data: Diccionario con los datos del balance
            
        Returns:
            Instancia de Balance
        """
        return cls(
            asset=data.get("asset", ""),
            free=data.get("free", 0.0),
            locked=data.get("locked", 0.0),
            total=data.get("total")
        )


class Transaction:
    """Clase para representar una transacción."""
    
    def __init__(
        self,
        transaction_id: Optional[str] = None,
        timestamp: Optional[Union[int, datetime]] = None,
        transaction_type: str = "",
        asset: str = "",
        amount: float = 0.0,
        price: Optional[float] = None,
        fee: Optional[float] = None,
        fee_asset: Optional[str] = None,
        source: str = "",
        destination: Optional[str] = None,
        status: str = "completed",
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar transacción.
        
        Args:
            transaction_id: ID de la transacción
            timestamp: Timestamp de la transacción
            transaction_type: Tipo de transacción (deposit, withdrawal, trade, transfer)
            asset: Activo de la transacción
            amount: Cantidad
            price: Precio (para trades)
            fee: Comisión
            fee_asset: Activo de la comisión
            source: Origen de la transacción
            destination: Destino de la transacción
            status: Estado (pending, completed, failed)
            notes: Notas adicionales
            metadata: Metadatos adicionales
        """
        self.transaction_id = transaction_id or str(uuid.uuid4())
        
        if timestamp is None:
            self.timestamp = datetime.now()
        elif isinstance(timestamp, int):
            self.timestamp = datetime.fromtimestamp(timestamp)
        else:
            self.timestamp = timestamp
        
        self.transaction_type = transaction_type
        self.asset = asset
        self.amount = amount
        self.price = price
        self.fee = fee
        self.fee_asset = fee_asset
        self.source = source
        self.destination = destination
        self.status = status
        self.notes = notes
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos de la transacción
        """
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "transaction_type": self.transaction_type,
            "asset": self.asset,
            "amount": self.amount,
            "price": self.price,
            "fee": self.fee,
            "fee_asset": self.fee_asset,
            "source": self.source,
            "destination": self.destination,
            "status": self.status,
            "notes": self.notes,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """
        Crear desde diccionario.
        
        Args:
            data: Diccionario con los datos de la transacción
            
        Returns:
            Instancia de Transaction
        """
        # Convertir timestamp
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            transaction_id=data.get("transaction_id"),
            timestamp=timestamp,
            transaction_type=data.get("transaction_type", ""),
            asset=data.get("asset", ""),
            amount=data.get("amount", 0.0),
            price=data.get("price"),
            fee=data.get("fee"),
            fee_asset=data.get("fee_asset"),
            source=data.get("source", ""),
            destination=data.get("destination"),
            status=data.get("status", "completed"),
            notes=data.get("notes"),
            metadata=data.get("metadata", {})
        )


class Account:
    """Clase para representar una cuenta o exchange."""
    
    def __init__(
        self,
        name: str,
        account_type: str = "exchange",
        active: bool = True,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar cuenta.
        
        Args:
            name: Nombre de la cuenta
            account_type: Tipo de cuenta (exchange, wallet, bank)
            active: Si la cuenta está activa
            description: Descripción de la cuenta
            metadata: Metadatos adicionales
        """
        self.name = name
        self.account_type = account_type
        self.active = active
        self.description = description
        self.metadata = metadata or {}
        
        # Balances por activo
        self.balances: Dict[str, Balance] = {}
    
    def update_balance(self, balance: Balance) -> None:
        """
        Actualizar balance de un activo.
        
        Args:
            balance: Balance a actualizar
        """
        self.balances[balance.asset] = balance
    
    def get_balance(self, asset: str) -> Optional[Balance]:
        """
        Obtener balance de un activo.
        
        Args:
            asset: Símbolo del activo
            
        Returns:
            Balance del activo, o None si no existe
        """
        return self.balances.get(asset)
    
    def get_total_balance_usd(self, asset_prices: Dict[str, float]) -> float:
        """
        Obtener balance total en USD.
        
        Args:
            asset_prices: Precios de los activos en USD
            
        Returns:
            Balance total en USD
        """
        total = 0.0
        
        for asset, balance in self.balances.items():
            # Obtener precio del activo (default 0)
            price = asset_prices.get(asset, 0.0)
            
            # Sumar al total
            total += balance.total * price
        
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con los datos de la cuenta
        """
        return {
            "name": self.name,
            "account_type": self.account_type,
            "active": self.active,
            "description": self.description,
            "metadata": self.metadata,
            "balances": {asset: balance.to_dict() for asset, balance in self.balances.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Account':
        """
        Crear desde diccionario.
        
        Args:
            data: Diccionario con los datos de la cuenta
            
        Returns:
            Instancia de Account
        """
        account = cls(
            name=data.get("name", ""),
            account_type=data.get("account_type", "exchange"),
            active=data.get("active", True),
            description=data.get("description"),
            metadata=data.get("metadata", {})
        )
        
        # Cargar balances
        balances_data = data.get("balances", {})
        for asset, balance_data in balances_data.items():
            account.balances[asset] = Balance.from_dict(balance_data)
        
        return account


class BalanceManager(Component):
    """
    Gestor de balances del sistema.
    
    Este componente se encarga de gestionar y monitorear los balances
    de diferentes cuentas, exchanges y wallets.
    """
    
    def __init__(self, name: str = "balance_manager"):
        """
        Inicializar el gestor de balances.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Cuentas gestionadas
        self.accounts: Dict[str, Account] = {}
        
        # Historial de transacciones
        self.transactions: List[Transaction] = []
        
        # Cache de precios de activos
        self.asset_prices: Dict[str, float] = {}
        
        # Directorio de datos
        self.data_dir = "data/accounting"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Límite de cache de transacciones
        self.max_transactions = 1000
    
    async def start(self) -> None:
        """Iniciar el gestor de balances."""
        await super().start()
        
        # Cargar datos
        await self._load_data()
        
        # Iniciar tareas periódicas
        self.update_task = asyncio.create_task(self._update_loop())
        
        self.logger.info("Gestor de balances iniciado")
    
    async def stop(self) -> None:
        """Detener el gestor de balances."""
        # Cancelar tareas
        if hasattr(self, 'update_task'):
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Guardar datos
        await self._save_data()
        
        await super().stop()
        self.logger.info("Gestor de balances detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Actualizar precios
        if event_type == "market.ticker_updated":
            symbol = data.get("symbol")
            price = data.get("price")
            
            if symbol and price:
                # Extraer activo base
                base_asset = symbol.split('/')[0] if '/' in symbol else symbol
                
                # Actualizar precio
                self.asset_prices[base_asset] = price
        
        # Procesar transacciones
        elif event_type == "transaction.created":
            await self._process_transaction(data)
        
        # Actualizar balance de cuenta específica
        elif event_type == "account.balance_updated":
            await self._update_account_balance(
                data.get("account_name"),
                data.get("balances", [])
            )
    
    async def add_account(self, account: Account) -> bool:
        """
        Añadir o actualizar una cuenta.
        
        Args:
            account: Cuenta a añadir
            
        Returns:
            True si se añadió correctamente, False en caso contrario
        """
        try:
            self.accounts[account.name] = account
            self.logger.info(f"Cuenta añadida/actualizada: {account.name}")
            
            # Guardar datos
            await self._save_data()
            
            return True
        except Exception as e:
            self.logger.error(f"Error al añadir cuenta: {e}")
            return False
    
    async def get_account(self, account_name: str) -> Optional[Account]:
        """
        Obtener una cuenta por su nombre.
        
        Args:
            account_name: Nombre de la cuenta
            
        Returns:
            Cuenta, o None si no existe
        """
        return self.accounts.get(account_name)
    
    async def get_accounts(self) -> List[Account]:
        """
        Obtener todas las cuentas.
        
        Returns:
            Lista de cuentas
        """
        return list(self.accounts.values())
    
    async def delete_account(self, account_name: str) -> bool:
        """
        Eliminar una cuenta.
        
        Args:
            account_name: Nombre de la cuenta
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if account_name in self.accounts:
            del self.accounts[account_name]
            self.logger.info(f"Cuenta eliminada: {account_name}")
            
            # Guardar datos
            await self._save_data()
            
            return True
        
        return False
    
    async def update_account_balance(
        self,
        account_name: str,
        asset: str,
        free: float,
        locked: float = 0.0,
        total: Optional[float] = None
    ) -> bool:
        """
        Actualizar balance de un activo en una cuenta.
        
        Args:
            account_name: Nombre de la cuenta
            asset: Símbolo del activo
            free: Cantidad disponible
            locked: Cantidad bloqueada
            total: Cantidad total
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        account = self.accounts.get(account_name)
        if not account:
            self.logger.error(f"Cuenta no encontrada: {account_name}")
            return False
        
        # Crear y actualizar balance
        balance = Balance(asset, free, locked, total)
        account.update_balance(balance)
        
        self.logger.info(f"Balance actualizado: {account_name} - {asset}")
        
        # Emitir evento
        await self.emit_event("account.balance_updated", {
            "account_name": account_name,
            "asset": asset,
            "balance": balance.to_dict()
        })
        
        return True
    
    async def add_transaction(self, transaction: Transaction) -> bool:
        """
        Añadir una transacción.
        
        Args:
            transaction: Transacción a añadir
            
        Returns:
            True si se añadió correctamente, False en caso contrario
        """
        try:
            # Validar transacción
            if not transaction.asset or not transaction.transaction_type:
                self.logger.error("Transacción inválida: faltan datos requeridos")
                return False
            
            # Añadir a la lista
            self.transactions.append(transaction)
            
            # Limitar tamaño
            if len(self.transactions) > self.max_transactions:
                self.transactions = self.transactions[-self.max_transactions:]
            
            self.logger.info(f"Transacción añadida: {transaction.transaction_id} - {transaction.transaction_type} - {transaction.asset}")
            
            # Procesar transacción para actualizar balances
            await self._process_transaction(transaction.to_dict())
            
            # Emitir evento
            await self.emit_event("transaction.added", {
                "transaction": transaction.to_dict()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Error al añadir transacción: {e}")
            return False
    
    async def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account: Optional[str] = None,
        asset: Optional[str] = None,
        transaction_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Transaction]:
        """
        Obtener transacciones filtradas.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            account: Filtrar por cuenta
            asset: Filtrar por activo
            transaction_type: Filtrar por tipo
            limit: Límite de resultados
            
        Returns:
            Lista de transacciones
        """
        # Filtrar transacciones
        filtered = self.transactions
        
        if start_date:
            filtered = [t for t in filtered if t.timestamp >= start_date]
        
        if end_date:
            filtered = [t for t in filtered if t.timestamp <= end_date]
        
        if account:
            filtered = [t for t in filtered if t.source == account or t.destination == account]
        
        if asset:
            filtered = [t for t in filtered if t.asset == asset]
        
        if transaction_type:
            filtered = [t for t in filtered if t.transaction_type == transaction_type]
        
        # Ordenar por fecha (más recientes primero)
        filtered.sort(key=lambda t: t.timestamp, reverse=True)
        
        # Limitar resultados
        return filtered[:limit]
    
    async def get_portfolio_value(self) -> Dict[str, Any]:
        """
        Obtener valor total del portfolio en USD.
        
        Returns:
            Información del portfolio
        """
        total_value = 0.0
        account_values = {}
        asset_distribution = {}
        
        # Calcular valor por cuenta
        for name, account in self.accounts.items():
            if not account.active:
                continue
            
            account_value = account.get_total_balance_usd(self.asset_prices)
            account_values[name] = account_value
            total_value += account_value
            
            # Calcular distribución por activo
            for asset, balance in account.balances.items():
                if balance.total <= 0:
                    continue
                
                asset_value = balance.total * self.asset_prices.get(asset, 0.0)
                
                if asset not in asset_distribution:
                    asset_distribution[asset] = 0.0
                
                asset_distribution[asset] += asset_value
        
        # Calcular porcentajes
        account_percentages = {name: safe_divide(value, total_value) for name, value in account_values.items()}
        asset_percentages = {asset: safe_divide(value, total_value) for asset, value in asset_distribution.items()}
        
        return {
            "total_value_usd": total_value,
            "account_values": account_values,
            "account_percentages": account_percentages,
            "asset_distribution": asset_distribution,
            "asset_percentages": asset_percentages,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_asset_balance(self, asset: str) -> Dict[str, Any]:
        """
        Obtener balance total de un activo en todas las cuentas.
        
        Args:
            asset: Símbolo del activo
            
        Returns:
            Información del balance
        """
        total = 0.0
        free = 0.0
        locked = 0.0
        by_account = {}
        
        # Sumar balances de todas las cuentas
        for name, account in self.accounts.items():
            if not account.active:
                continue
            
            balance = account.get_balance(asset)
            if balance:
                by_account[name] = balance.to_dict()
                total += balance.total
                free += balance.free
                locked += balance.locked
        
        # Calcular valor en USD
        price_usd = self.asset_prices.get(asset, 0.0)
        value_usd = total * price_usd
        
        return {
            "asset": asset,
            "total": total,
            "free": free,
            "locked": locked,
            "price_usd": price_usd,
            "value_usd": value_usd,
            "by_account": by_account,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_transaction(self, data: Dict[str, Any]) -> None:
        """
        Procesar una transacción para actualizar balances.
        
        Args:
            data: Datos de la transacción
        """
        # Convertir a objeto Transaction si es un diccionario
        transaction = data if isinstance(data, Transaction) else Transaction.from_dict(data)
        
        # Actualizar balances según tipo de transacción
        if transaction.transaction_type == "deposit":
            # Depósito: aumenta el balance en la cuenta destino
            if transaction.destination:
                account = self.accounts.get(transaction.destination)
                if account:
                    balance = account.get_balance(transaction.asset) or Balance(transaction.asset)
                    
                    # Actualizar balance
                    new_free = balance.free + transaction.amount
                    
                    # Actualizar en la cuenta
                    await self.update_account_balance(
                        transaction.destination,
                        transaction.asset,
                        new_free,
                        balance.locked
                    )
        
        elif transaction.transaction_type == "withdrawal":
            # Retiro: disminuye el balance en la cuenta origen
            if transaction.source:
                account = self.accounts.get(transaction.source)
                if account:
                    balance = account.get_balance(transaction.asset)
                    if balance:
                        # Actualizar balance
                        new_free = max(0, balance.free - transaction.amount)
                        
                        # Actualizar en la cuenta
                        await self.update_account_balance(
                            transaction.source,
                            transaction.asset,
                            new_free,
                            balance.locked
                        )
        
        elif transaction.transaction_type == "transfer":
            # Transferencia: disminuye en origen y aumenta en destino
            # Origen
            if transaction.source:
                source_account = self.accounts.get(transaction.source)
                if source_account:
                    source_balance = source_account.get_balance(transaction.asset)
                    if source_balance:
                        # Actualizar balance
                        new_free = max(0, source_balance.free - transaction.amount)
                        
                        # Actualizar en la cuenta
                        await self.update_account_balance(
                            transaction.source,
                            transaction.asset,
                            new_free,
                            source_balance.locked
                        )
            
            # Destino
            if transaction.destination:
                dest_account = self.accounts.get(transaction.destination)
                if dest_account:
                    dest_balance = dest_account.get_balance(transaction.asset) or Balance(transaction.asset)
                    
                    # Actualizar balance
                    new_free = dest_balance.free + transaction.amount
                    
                    # Actualizar en la cuenta
                    await self.update_account_balance(
                        transaction.destination,
                        transaction.asset,
                        new_free,
                        dest_balance.locked
                    )
        
        elif transaction.transaction_type == "trade":
            # Trade: intercambio entre dos activos
            if transaction.source and "metadata" in data:
                metadata = data["metadata"]
                
                # Verificar si hay información de trade
                base_asset = metadata.get("base_asset")
                quote_asset = metadata.get("quote_asset")
                base_amount = metadata.get("base_amount", 0.0)
                quote_amount = metadata.get("quote_amount", 0.0)
                
                if base_asset and quote_asset and transaction.source:
                    account = self.accounts.get(transaction.source)
                    if account:
                        # Actualizar base asset (lo que se vende/compra)
                        base_balance = account.get_balance(base_asset) or Balance(base_asset)
                        new_base_free = base_balance.free + base_amount  # Puede ser negativo en una venta
                        
                        # Actualizar quote asset (la moneda con la que se paga)
                        quote_balance = account.get_balance(quote_asset) or Balance(quote_asset)
                        new_quote_free = quote_balance.free + quote_amount  # Negativo en una compra
                        
                        # Actualizar balances
                        await self.update_account_balance(
                            transaction.source,
                            base_asset,
                            max(0, new_base_free),
                            base_balance.locked
                        )
                        
                        await self.update_account_balance(
                            transaction.source,
                            quote_asset,
                            max(0, new_quote_free),
                            quote_balance.locked
                        )
    
    async def _update_account_balance(self, account_name: str, balances: List[Dict[str, Any]]) -> None:
        """
        Actualizar múltiples balances de una cuenta.
        
        Args:
            account_name: Nombre de la cuenta
            balances: Lista de balances a actualizar
        """
        if not account_name or account_name not in self.accounts:
            self.logger.error(f"Cuenta no encontrada: {account_name}")
            return
        
        for balance_data in balances:
            asset = balance_data.get("asset")
            free = balance_data.get("free", 0.0)
            locked = balance_data.get("locked", 0.0)
            total = balance_data.get("total")
            
            if asset:
                await self.update_account_balance(account_name, asset, free, locked, total)
    
    async def _update_loop(self) -> None:
        """Bucle de actualización periódica."""
        while True:
            try:
                # Obtener datos actualizados de los exchanges
                await self._fetch_exchange_balances()
                
                # Guardar datos
                await self._save_data()
                
                # Calcular y emitir estado del portfolio
                portfolio = await self.get_portfolio_value()
                await self.emit_event("portfolio.updated", portfolio)
                
            except asyncio.CancelledError:
                # El bucle fue cancelado
                break
            except Exception as e:
                self.logger.error(f"Error en bucle de actualización: {e}")
            
            # Esperar para la próxima actualización (cada 15 minutos)
            await asyncio.sleep(900)
    
    async def _fetch_exchange_balances(self) -> None:
        """Obtener balances actualizados de los exchanges."""
        for account_name, account in self.accounts.items():
            if not account.active or account.account_type != "exchange":
                continue
            
            try:
                # Buscar un exchange connector en el bus de eventos
                exchange_data = {"account_name": account_name}
                
                # Solicitar balance al exchange
                await self.emit_event("exchange.fetch_balance", exchange_data)
                
                # NOTA: El exchange debería responder con un evento "account.balance_updated"
                # que será manejado por el método handle_event
                
            except Exception as e:
                self.logger.error(f"Error al obtener balance del exchange {account_name}: {e}")
    
    async def _load_data(self) -> None:
        """Cargar datos desde archivos."""
        try:
            # Cargar cuentas
            accounts_path = f"{self.data_dir}/accounts.json"
            accounts_data = load_json_file(accounts_path, {})
            
            if accounts_data:
                for name, data in accounts_data.items():
                    self.accounts[name] = Account.from_dict(data)
                
                self.logger.info(f"Cargadas {len(self.accounts)} cuentas")
            
            # Cargar transacciones
            transactions_path = f"{self.data_dir}/transactions.json"
            transactions_data = load_json_file(transactions_path, [])
            
            if transactions_data:
                self.transactions = [Transaction.from_dict(t) for t in transactions_data]
                
                # Limitar cantidad
                if len(self.transactions) > self.max_transactions:
                    self.transactions = self.transactions[-self.max_transactions:]
                
                self.logger.info(f"Cargadas {len(self.transactions)} transacciones")
        
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {e}")
    
    async def _save_data(self) -> None:
        """Guardar datos a archivos."""
        try:
            # Guardar cuentas
            accounts_data = {name: account.to_dict() for name, account in self.accounts.items()}
            accounts_path = f"{self.data_dir}/accounts.json"
            
            if save_json_file(accounts_path, accounts_data):
                self.logger.debug(f"Guardadas {len(self.accounts)} cuentas")
            
            # Guardar transacciones
            transactions_data = [t.to_dict() for t in self.transactions]
            transactions_path = f"{self.data_dir}/transactions.json"
            
            if save_json_file(transactions_path, transactions_data):
                self.logger.debug(f"Guardadas {len(self.transactions)} transacciones")
        
        except Exception as e:
            self.logger.error(f"Error al guardar datos: {e}")


# Exportación para uso fácil
balance_manager = BalanceManager()