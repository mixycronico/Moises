"""
Gestor de Inversionistas para el Sistema Genesis

Este módulo implementa la gestión de inversionistas dentro del Sistema Genesis,
permitiendo un manejo transparente y ético de las relaciones financieras entre
todos los participantes, siguiendo el principio fundamental:
"Todos ganamos o todos perdemos".

Autor: Genesis AI Assistant
Versión: 1.0.0 (Divina)
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import uuid
import json

from genesis.core.base import Component
from genesis.utils.helpers import generate_id, format_timestamp
from genesis.utils.logger import setup_logging
from genesis.db.transcendental_database import TranscendentalDatabase

# Configurar precisión para operaciones con Decimal
getcontext().prec = 28

# Estados de inversionista
class InvestorStatus:
    """Estados posibles para un inversionista."""
    ACTIVE = "active"         # Inversionista activo
    SUSPENDED = "suspended"   # Temporalmente suspendido
    INACTIVE = "inactive"     # Inactivo (por decisión propia)
    PENDING = "pending"       # Pendiente de aprobación/verificación
    
# Tipos de transacciones
class TransactionType:
    """Tipos de transacciones financieras en el sistema."""
    DEPOSIT = "deposit"               # Depósito de capital
    WITHDRAWAL = "withdrawal"         # Retiro de capital
    PROFIT_DISTRIBUTION = "profit"    # Distribución de ganancias
    TRANSFER = "transfer"             # Transferencia entre inversionistas
    BONUS = "bonus"                   # Bono por rendimiento
    LOAN = "loan"                     # Préstamo de capital
    LOAN_PAYMENT = "loan_payment"     # Pago de préstamo
    COMMISSION = "commission"         # Comisión para administrador
    FEE = "fee"                       # Comisión del sistema
    MAINTENANCE_COLLECTION = "maintenance_collection"  # Recolección para fondo de mantenimiento
    MAINTENANCE_USAGE = "maintenance_usage"           # Uso de fondos de mantenimiento
    ANNUAL_DISTRIBUTION = "annual_distribution"       # Distribución anual del excedente

# Tipos de auditoría
class AuditActionType:
    """Tipos de acciones para auditoría."""
    MAINTENANCE_FUND_COLLECTION = "maintenance_collection"  # Recolección para fondo de mantenimiento
    MAINTENANCE_FUND_USAGE = "maintenance_usage"            # Uso de fondos de mantenimiento
    ANNUAL_DISTRIBUTION = "annual_distribution"             # Distribución anual del excedente
    SYSTEM_CONFIG_CHANGE = "system_config_change"           # Cambio en configuración del sistema
    INVESTOR_STATUS_CHANGE = "investor_status_change"       # Cambio en estado de inversionista
    ADMIN_ACTION = "admin_action"                          # Acción administrativa general

class InvestorManager:
    """
    Gestor de inversionistas para el Sistema Genesis.
    
    Esta clase implementa todas las funciones necesarias para gestionar 
    inversionistas, incluyendo sus balances, transacciones y relaciones.
    """
    
    def __init__(self):
        """Inicializar el gestor de inversionistas."""
        self.logger = logging.getLogger(__name__)
        self.db = None
        self.investors = {}
        self.transactions = {}
        self.initialized = False
        self.total_capital = Decimal('0')
        self.daily_stats = {
            "bonus_enabled": True,
            "loans_enabled": True,
            "commissions_allowed": True,
            "transfer_limit": Decimal('200'),
            "notes": ""
        }
        self.referral_contracts = {}
        self.active_loans = {}
        self.performance_bonuses = {}
        
        # Fondo de mantenimiento (5% semanal)
        self.maintenance_fund = Decimal('0')
        self.last_maintenance_collection = None
        self.annual_distribution_date = None
        
        # Control para la devolución especial de $1500 (10% semanal después del primer mes)
        self.founder_repayment = {
            "total_amount": Decimal('1500'),
            "remaining": Decimal('1500'),
            "started": False,
            "start_date": None,
            "last_payment": None,
            "finished": False,
            "investor_id": "moises"  # ID del fundador (tu ID)
        }
        
        self.logger.info("InvestorManager inicializado")
        
    async def initialize(self, db: Optional[TranscendentalDatabase] = None) -> bool:
        """
        Inicializar el gestor con conexión a base de datos.
        
        Args:
            db: Conexión a base de datos (opcional)
            
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            self.db = db
            
            # Cargar inversionistas y transacciones desde la base de datos
            if self.db:
                await self._load_data_from_db()
            else:
                self.logger.warning("No se proporcionó conexión a base de datos, usando almacenamiento en memoria")
                
            # Inicializar estadísticas diarias
            await self._initialize_daily_stats()
            
            # Verificar y ejecutar recolección semanal para fondo de mantenimiento si es necesario
            await self._check_maintenance_fund_collection()
            
            # Verificar y ejecutar distribución anual si es necesario
            await self._check_annual_distribution()
                
            self.initialized = True
            self.logger.info(f"InvestorManager inicializado con {len(self.investors)} inversionistas")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al inicializar InvestorManager: {str(e)}")
            return False
            
    async def _load_data_from_db(self) -> None:
        """Cargar datos desde la base de datos."""
        try:
            if not self.db:
                return
                
            # Cargar inversionistas
            investor_data = await self.db.get_all("investor:")
            if investor_data:
                for key, value in investor_data.items():
                    investor_id = key.split(":", 1)[1]
                    self.investors[investor_id] = json.loads(value)
                
            # Cargar transacciones
            transaction_data = await self.db.get_all("transaction:")
            if transaction_data:
                for key, value in transaction_data.items():
                    transaction_id = key.split(":", 1)[1]
                    self.transactions[transaction_id] = json.loads(value)
                
            # Cargar contratos de referidos
            contract_data = await self.db.get_all("referral_contract:")
            if contract_data:
                for key, value in contract_data.items():
                    contract_id = key.split(":", 1)[1]
                    self.referral_contracts[contract_id] = json.loads(value)
                
            # Cargar préstamos activos
            loan_data = await self.db.get_all("loan:")
            if loan_data:
                for key, value in loan_data.items():
                    loan_id = key.split(":", 1)[1]
                    self.active_loans[loan_id] = json.loads(value)
                
            # Cargar bonos de rendimiento
            bonus_data = await self.db.get_all("performance_bonus:")
            if bonus_data:
                for key, value in bonus_data.items():
                    bonus_id = key.split(":", 1)[1]
                    self.performance_bonuses[bonus_id] = json.loads(value)
            
            # Cargar información del fondo de mantenimiento
            maintenance_data = await self.db.get("system:maintenance_fund")
            if maintenance_data:
                maintenance_info = json.loads(maintenance_data)
                self.maintenance_fund = Decimal(str(maintenance_info.get("balance", 0)))
                self.last_maintenance_collection = maintenance_info.get("last_collection")
                self.annual_distribution_date = maintenance_info.get("annual_distribution_date")
                
            # Calcular capital total
            self.total_capital = Decimal('0')
            for investor_id, investor in self.investors.items():
                self.total_capital += Decimal(str(investor.get("balance", 0)))
                
            self.logger.info(f"Datos cargados desde DB: {len(self.investors)} inversionistas, {len(self.transactions)} transacciones")
            
        except Exception as e:
            self.logger.error(f"Error al cargar datos desde DB: {str(e)}")
    
    async def _initialize_daily_stats(self) -> None:
        """Inicializar estadísticas diarias."""
        try:
            if self.db:
                daily_stats = await self.db.get("system:daily_stats")
                if daily_stats:
                    self.daily_stats = json.loads(daily_stats)
                else:
                    # Crear estadísticas iniciales y guardar
                    await self.db.set("system:daily_stats", json.dumps(self.daily_stats))
            
            self.logger.info(f"Estadísticas diarias inicializadas: bonos {self.daily_stats['bonus_enabled']}, préstamos {self.daily_stats['loans_enabled']}")
            
        except Exception as e:
            self.logger.error(f"Error al inicializar estadísticas diarias: {str(e)}")
    
    async def create_investor(self, 
                             investor_id: str,
                             name: str,
                             email: str,
                             initial_balance: float = 0.0,
                             admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Crear un nuevo inversionista en el sistema.
        
        Args:
            investor_id: ID único del inversionista
            name: Nombre completo
            email: Correo electrónico
            initial_balance: Saldo inicial (opcional)
            admin_id: ID del administrador que lo crea (opcional)
            
        Returns:
            Datos del inversionista creado
        """
        try:
            # Verificar si ya existe
            if investor_id in self.investors:
                return {"success": False, "error": "El inversionista ya existe"}
                
            # Crear registro de inversionista
            timestamp = datetime.now().isoformat()
            investor = {
                "id": investor_id,
                "name": name,
                "email": email,
                "balance": Decimal(str(initial_balance)),
                "invested_capital": Decimal(str(initial_balance)),
                "total_profit": Decimal('0'),
                "status": InvestorStatus.ACTIVE,
                "created_at": timestamp,
                "updated_at": timestamp,
                "created_by": admin_id,
                "last_distribution": None,
                "performance_stats": {
                    "total_bonus": Decimal('0'),
                    "total_loans": Decimal('0'),
                    "active_loans": Decimal('0'),
                    "total_commissions": Decimal('0')
                }
            }
            
            # Guardar en memoria
            self.investors[investor_id] = investor
            
            # Actualizar capital total
            self.total_capital += Decimal(str(initial_balance))
            
            # Si hay saldo inicial, registrar como depósito
            if initial_balance > 0:
                await self._register_transaction(
                    investor_id=investor_id,
                    transaction_type=TransactionType.DEPOSIT,
                    amount=initial_balance,
                    description="Depósito inicial",
                    related_id=None,
                    admin_id=admin_id
                )
            
            # Guardar en base de datos
            if self.db:
                await self.db.set(f"investor:{investor_id}", json.dumps(investor))
                
            self.logger.info(f"Inversionista creado: {investor_id}, balance inicial: ${initial_balance}")
            
            return {"success": True, "investor": self._clean_for_response(investor)}
            
        except Exception as e:
            self.logger.error(f"Error al crear inversionista: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def update_investor(self,
                             investor_id: str,
                             status: Optional[str] = None,
                             email: Optional[str] = None,
                             name: Optional[str] = None,
                             admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Actualizar datos de un inversionista.
        
        Args:
            investor_id: ID del inversionista
            status: Nuevo estado (opcional)
            email: Nuevo email (opcional)
            name: Nuevo nombre (opcional)
            admin_id: ID del administrador que realiza el cambio
            
        Returns:
            Resultado de la actualización
        """
        try:
            # Verificar si existe
            if investor_id not in self.investors:
                return {"success": False, "error": "Inversionista no encontrado"}
                
            investor = self.investors[investor_id]
            modified = False
            
            # Actualizar campos si se proporcionan
            if status and status in [InvestorStatus.ACTIVE, InvestorStatus.SUSPENDED,
                                 InvestorStatus.INACTIVE, InvestorStatus.PENDING]:
                investor["status"] = status
                modified = True
                
            if email:
                investor["email"] = email
                modified = True
                
            if name:
                investor["name"] = name
                modified = True
                
            if modified:
                investor["updated_at"] = datetime.now().isoformat()
                
                # Guardar en base de datos
                if self.db:
                    await self.db.set(f"investor:{investor_id}", json.dumps(investor))
                    
                self.logger.info(f"Inversionista actualizado: {investor_id}")
                
                return {"success": True, "investor": self._clean_for_response(investor)}
            else:
                return {"success": True, "message": "No se realizaron cambios"}
                
        except Exception as e:
            self.logger.error(f"Error al actualizar inversionista: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_investor(self, investor_id: str) -> Dict[str, Any]:
        """
        Obtener datos de un inversionista.
        
        Args:
            investor_id: ID del inversionista
            
        Returns:
            Datos del inversionista
        """
        try:
            if investor_id not in self.investors:
                return {"success": False, "error": "Inversionista no encontrado"}
                
            investor = self.investors[investor_id]
            
            # Obtener transacciones recientes
            transactions = await self.get_transactions(investor_id, limit=10)
            
            # Obtener préstamos activos
            active_loans = await self.get_active_loans(investor_id)
            
            # Obtener contratos de referidos
            contracts = await self.get_referral_contracts(investor_id)
            
            # Construir respuesta completa
            response = {
                "success": True,
                "investor": self._clean_for_response(investor),
                "recent_transactions": transactions.get("transactions", []),
                "active_loans": active_loans.get("loans", []),
                "referral_contracts": contracts.get("contracts", [])
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error al obtener inversionista: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_all_investors(self) -> Dict[str, Any]:
        """
        Obtener lista de todos los inversionistas.
        
        Returns:
            Lista de inversionistas
        """
        try:
            investors_list = [
                self._clean_for_response(investor)
                for investor in self.investors.values()
            ]
            
            # Estadísticas globales
            stats = {
                "total_investors": len(investors_list),
                "active_investors": sum(1 for i in investors_list if i["status"] == InvestorStatus.ACTIVE),
                "total_capital": float(self.total_capital),
                "total_profit_distributed": sum(float(i["total_profit"]) for i in investors_list),
                "maintenance_fund": float(self.maintenance_fund),
                "daily_stats": self.daily_stats
            }
            
            return {
                "success": True,
                "investors": investors_list,
                "stats": stats
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener todos los inversionistas: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def deposit(self,
                     investor_id: str,
                     amount: float,
                     description: str = "Depósito de capital",
                     admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Realizar un depósito a un inversionista.
        
        Args:
            investor_id: ID del inversionista
            amount: Cantidad a depositar
            description: Descripción del depósito
            admin_id: ID del administrador que registra (opcional)
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si existe y está activo
            if investor_id not in self.investors:
                return {"success": False, "error": "Inversionista no encontrado"}
                
            investor = self.investors[investor_id]
            
            if investor["status"] != InvestorStatus.ACTIVE:
                return {"success": False, "error": "El inversionista no está activo"}
                
            if amount <= 0:
                return {"success": False, "error": "El monto debe ser mayor a cero"}
                
            # Actualizar saldo
            amount_decimal = Decimal(str(amount))
            investor["balance"] += amount_decimal
            investor["invested_capital"] += amount_decimal
            investor["updated_at"] = datetime.now().isoformat()
            
            # Actualizar capital total
            self.total_capital += amount_decimal
            
            # Registrar transacción
            transaction = await self._register_transaction(
                investor_id=investor_id,
                transaction_type=TransactionType.DEPOSIT,
                amount=amount,
                description=description,
                related_id=None,
                admin_id=admin_id
            )
            
            # Guardar cambios
            if self.db:
                await self.db.set(f"investor:{investor_id}", json.dumps(investor))
                
            self.logger.info(f"Depósito realizado: ${amount} para {investor_id}")
            
            return {
                "success": True,
                "investor": self._clean_for_response(investor),
                "transaction": transaction,
                "new_balance": float(investor["balance"])
            }
            
        except Exception as e:
            self.logger.error(f"Error al realizar depósito: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def withdraw(self,
                      investor_id: str,
                      amount: float,
                      description: str = "Retiro de capital",
                      admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Realizar un retiro de un inversionista.
        
        Args:
            investor_id: ID del inversionista
            amount: Cantidad a retirar
            description: Descripción del retiro
            admin_id: ID del administrador que registra (opcional)
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si existe y está activo
            if investor_id not in self.investors:
                return {"success": False, "error": "Inversionista no encontrado"}
                
            investor = self.investors[investor_id]
            
            if investor["status"] != InvestorStatus.ACTIVE:
                return {"success": False, "error": "El inversionista no está activo"}
                
            if amount <= 0:
                return {"success": False, "error": "El monto debe ser mayor a cero"}
                
            # Verificar saldo suficiente
            amount_decimal = Decimal(str(amount))
            if investor["balance"] < amount_decimal:
                return {"success": False, "error": "Saldo insuficiente"}
                
            # Actualizar saldo
            investor["balance"] -= amount_decimal
            investor["updated_at"] = datetime.now().isoformat()
            
            # Ajustar capital invertido si es necesario
            if investor["invested_capital"] > investor["balance"]:
                investor["invested_capital"] = investor["balance"]
                
            # Actualizar capital total
            self.total_capital -= amount_decimal
            
            # Registrar transacción
            transaction = await self._register_transaction(
                investor_id=investor_id,
                transaction_type=TransactionType.WITHDRAWAL,
                amount=-amount,  # Negativo para retiros
                description=description,
                related_id=None,
                admin_id=admin_id
            )
            
            # Guardar cambios
            if self.db:
                await self.db.set(f"investor:{investor_id}", json.dumps(investor))
                
            self.logger.info(f"Retiro realizado: ${amount} para {investor_id}")
            
            return {
                "success": True,
                "investor": self._clean_for_response(investor),
                "transaction": transaction,
                "new_balance": float(investor["balance"])
            }
            
        except Exception as e:
            self.logger.error(f"Error al realizar retiro: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def distribute_profit(self,
                               total_profit: float,
                               distribution_method: str = "proportional",
                               description: str = "Distribución de ganancias",
                               admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Distribuir ganancias entre inversionistas activos.
        
        Args:
            total_profit: Ganancia total a distribuir
            distribution_method: Método de distribución ("proportional", "equal")
            description: Descripción de la distribución
            admin_id: ID del administrador que registra (opcional)
            
        Returns:
            Resultado de la distribución
        """
        try:
            if total_profit <= 0:
                return {"success": False, "error": "La ganancia debe ser mayor a cero"}
                
            # Obtener inversionistas activos
            active_investors = {
                investor_id: investor
                for investor_id, investor in self.investors.items()
                if investor["status"] == InvestorStatus.ACTIVE
            }
            
            if not active_investors:
                return {"success": False, "error": "No hay inversionistas activos"}
                
            # Calcular distribución según método
            distribution = {}
            
            if distribution_method == "equal":
                # Distribución equitativa
                individual_profit = Decimal(str(total_profit)) / Decimal(str(len(active_investors)))
                
                for investor_id in active_investors:
                    distribution[investor_id] = individual_profit
                    
            else:  # Por defecto: proportional
                # Distribución proporcional al capital invertido
                total_invested = sum(investor["invested_capital"] for investor in active_investors.values())
                
                if total_invested <= 0:
                    # Si no hay capital invertido, distribuir equitativamente
                    individual_profit = Decimal(str(total_profit)) / Decimal(str(len(active_investors)))
                    
                    for investor_id in active_investors:
                        distribution[investor_id] = individual_profit
                else:
                    # Distribuir proporcionalmente
                    profit_decimal = Decimal(str(total_profit))
                    
                    for investor_id, investor in active_investors.items():
                        proportion = investor["invested_capital"] / total_invested
                        distribution[investor_id] = profit_decimal * proportion
            
            # Aplicar distribución
            transactions = []
            distribution_date = datetime.now().isoformat()
            
            for investor_id, amount in distribution.items():
                investor = self.investors[investor_id]
                
                # Actualizar saldo
                investor["balance"] += amount
                investor["total_profit"] += amount
                investor["updated_at"] = distribution_date
                investor["last_distribution"] = distribution_date
                
                # Registrar transacción
                transaction = await self._register_transaction(
                    investor_id=investor_id,
                    transaction_type=TransactionType.PROFIT_DISTRIBUTION,
                    amount=float(amount),
                    description=description,
                    related_id=None,
                    admin_id=admin_id
                )
                transactions.append(transaction)
                
                # Guardar cambios
                if self.db:
                    await self.db.set(f"investor:{investor_id}", json.dumps(investor))
            
            # Verificar si se aplican bonos por rendimiento
            if self.daily_stats["bonus_enabled"]:
                await self._apply_performance_bonuses()
                
            self.logger.info(f"Ganancias distribuidas: ${total_profit} entre {len(active_investors)} inversionistas")
            
            return {
                "success": True,
                "total_profit": total_profit,
                "distribution_method": distribution_method,
                "investors_count": len(distribution),
                "distribution": {k: float(v) for k, v in distribution.items()},
                "transactions": transactions
            }
            
        except Exception as e:
            self.logger.error(f"Error al distribuir ganancias: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def transfer(self,
                      from_investor_id: str,
                      to_investor_id: str,
                      amount: float,
                      reason: Optional[str] = None,
                      admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Realizar transferencia entre inversionistas.
        
        Args:
            from_investor_id: ID del inversionista que envía
            to_investor_id: ID del inversionista que recibe
            amount: Cantidad a transferir
            reason: Motivo de la transferencia (opcional)
            admin_id: ID del administrador que autoriza (opcional)
            
        Returns:
            Resultado de la transferencia
        """
        try:
            # Validar parámetros
            if from_investor_id == to_investor_id:
                return {"success": False, "error": "No se puede transferir a sí mismo"}
                
            if amount <= 0:
                return {"success": False, "error": "El monto debe ser mayor a cero"}
                
            # Verificar límite diario de transferencia
            transfer_limit = float(self.daily_stats["transfer_limit"])
            if amount > transfer_limit and not admin_id:
                return {
                    "success": False, 
                    "error": f"El monto excede el límite diario de transferencia (${transfer_limit})"
                }
                
            # Verificar inversionistas
            if from_investor_id not in self.investors:
                return {"success": False, "error": "Inversionista emisor no encontrado"}
                
            if to_investor_id not in self.investors:
                return {"success": False, "error": "Inversionista receptor no encontrado"}
                
            sender = self.investors[from_investor_id]
            recipient = self.investors[to_investor_id]
            
            # Verificar estado activo
            if sender["status"] != InvestorStatus.ACTIVE:
                return {"success": False, "error": "El inversionista emisor no está activo"}
                
            if recipient["status"] != InvestorStatus.ACTIVE:
                return {"success": False, "error": "El inversionista receptor no está activo"}
                
            # Verificar saldo suficiente
            amount_decimal = Decimal(str(amount))
            if sender["balance"] < amount_decimal:
                return {"success": False, "error": "Saldo insuficiente para realizar la transferencia"}
                
            # Realizar transferencia
            transfer_id = generate_id()
            timestamp = datetime.now().isoformat()
            description = f"Transferencia a {recipient['name']}"
            recipient_description = f"Transferencia de {sender['name']}"
            
            if reason:
                description += f" - {reason}"
                recipient_description += f" - {reason}"
                
            # Actualizar saldos
            sender["balance"] -= amount_decimal
            sender["updated_at"] = timestamp
            
            recipient["balance"] += amount_decimal
            recipient["updated_at"] = timestamp
            
            # Registrar transacciones
            sender_transaction = await self._register_transaction(
                investor_id=from_investor_id,
                transaction_type=TransactionType.TRANSFER,
                amount=-float(amount_decimal),
                description=description,
                related_id=to_investor_id,
                admin_id=admin_id,
                metadata={"transfer_id": transfer_id}
            )
            
            recipient_transaction = await self._register_transaction(
                investor_id=to_investor_id,
                transaction_type=TransactionType.TRANSFER,
                amount=float(amount_decimal),
                description=recipient_description,
                related_id=from_investor_id,
                admin_id=admin_id,
                metadata={"transfer_id": transfer_id}
            )
            
            # Guardar cambios
            if self.db:
                await self.db.set(f"investor:{from_investor_id}", json.dumps(sender))
                await self.db.set(f"investor:{to_investor_id}", json.dumps(recipient))
                
                # Guardar transferencia
                transfer_data = {
                    "id": transfer_id,
                    "emisor": from_investor_id,
                    "receptor": to_investor_id,
                    "monto": float(amount_decimal),
                    "motivo": reason,
                    "fecha": timestamp,
                    "estado": "completada"
                }
                
                await self.db.set(f"transfer:{transfer_id}", json.dumps(transfer_data))
                
            self.logger.info(f"Transferencia realizada: ${amount} de {from_investor_id} a {to_investor_id}")
            
            return {
                "success": True,
                "transfer_id": transfer_id,
                "amount": float(amount_decimal),
                "sender": self._clean_for_response(sender),
                "recipient": self._clean_for_response(recipient),
                "sender_transaction": sender_transaction,
                "recipient_transaction": recipient_transaction
            }
            
        except Exception as e:
            self.logger.error(f"Error al realizar transferencia: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def create_loan(self,
                         investor_id: str,
                         amount: float,
                         reinvest: bool = False,
                         admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Crear préstamo por capital para un inversionista.
        
        Args:
            investor_id: ID del inversionista
            amount: Cantidad del préstamo
            reinvest: Si el préstamo se reinvierte automáticamente
            admin_id: ID del administrador que autoriza
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si los préstamos están habilitados
            if not self.daily_stats["loans_enabled"]:
                return {"success": False, "error": "Los préstamos están deshabilitados temporalmente"}
                
            # Verificar inversionista
            if investor_id not in self.investors:
                return {"success": False, "error": "Inversionista no encontrado"}
                
            investor = self.investors[investor_id]
            
            # Verificar estado activo
            if investor["status"] != InvestorStatus.ACTIVE:
                return {"success": False, "error": "El inversionista no está activo"}
                
            # Verificar elegibilidad (al menos 3 meses de antigüedad)
            created_date = datetime.fromisoformat(investor["created_at"])
            min_tenure = timedelta(days=90)  # 3 meses
            
            if datetime.now() - created_date < min_tenure:
                days_remaining = min_tenure - (datetime.now() - created_date)
                return {
                    "success": False, 
                    "error": f"El inversionista no cumple con la antigüedad mínima. Faltan {days_remaining.days} días."
                }
                
            # Calcular capital elegible (40% del capital invertido)
            invested_capital = float(investor["invested_capital"])
            eligible_capital = invested_capital * 0.4
            
            if amount > eligible_capital:
                return {
                    "success": False,
                    "error": f"El monto solicitado excede el capital elegible (40% de ${invested_capital} = ${eligible_capital:.2f})"
                }
                
            # Verificar si ya tiene préstamos activos
            active_investor_loans = [
                loan for loan in self.active_loans.values()
                if loan["inversionista"] == investor_id and loan["estado"] == "activo"
            ]
            
            if active_investor_loans:
                return {
                    "success": False,
                    "error": "El inversionista ya tiene un préstamo activo"
                }
                
            # Crear préstamo
            loan_id = generate_id()
            timestamp = datetime.now().isoformat()
            
            loan = {
                "id": loan_id,
                "inversionista": investor_id,
                "capitalElegible": eligible_capital,
                "montoPrestado": amount,
                "fechaInicio": timestamp,
                "estado": "activo",
                "porcentajeRetencion": 30,  # 30% de la ganancia diaria
                "saldoPendiente": amount,
                "reinvertido": reinvest,
                "pagosRealizados": [],
                "ultimoPago": None,
                "aprobadoPor": admin_id
            }
            
            # Registrar préstamo
            self.active_loans[loan_id] = loan
            
            # Actualizar estadísticas del inversionista
            investor["performance_stats"]["total_loans"] += Decimal(str(amount))
            investor["performance_stats"]["active_loans"] += Decimal(str(amount))
            
            # Si no se reinvierte, añadir al saldo del inversionista
            if not reinvest:
                investor["balance"] += Decimal(str(amount))
                
            investor["updated_at"] = timestamp
            
            # Registrar transacción
            transaction = await self._register_transaction(
                investor_id=investor_id,
                transaction_type=TransactionType.LOAN,
                amount=amount,
                description=f"Préstamo por capital ({reinvest and 'reinvertido' or 'retirado'})",
                related_id=None,
                admin_id=admin_id,
                metadata={"loan_id": loan_id, "reinvested": reinvest}
            )
            
            # Guardar cambios
            if self.db:
                await self.db.set(f"loan:{loan_id}", json.dumps(loan))
                await self.db.set(f"investor:{investor_id}", json.dumps(investor))
                
            self.logger.info(f"Préstamo creado: ${amount} para {investor_id} ({reinvest and 'reinvertido' or 'retirado'})")
            
            return {
                "success": True,
                "loan": loan,
                "investor": self._clean_for_response(investor),
                "transaction": transaction
            }
            
        except Exception as e:
            self.logger.error(f"Error al crear préstamo: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def process_loan_payment(self,
                                  loan_id: str,
                                  amount: float,
                                  admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Procesar pago de un préstamo.
        
        Args:
            loan_id: ID del préstamo
            amount: Cantidad a pagar
            admin_id: ID del administrador que registra
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar préstamo
            if loan_id not in self.active_loans:
                return {"success": False, "error": "Préstamo no encontrado"}
                
            loan = self.active_loans[loan_id]
            
            # Verificar estado
            if loan["estado"] != "activo":
                return {"success": False, "error": "El préstamo no está activo"}
                
            # Verificar monto
            if amount <= 0:
                return {"success": False, "error": "El monto debe ser mayor a cero"}
                
            if amount > loan["saldoPendiente"]:
                return {"success": False, "error": "El monto excede el saldo pendiente"}
                
            # Actualizar préstamo
            timestamp = datetime.now().isoformat()
            loan["saldoPendiente"] -= amount
            loan["ultimoPago"] = timestamp
            
            payment = {
                "fecha": timestamp,
                "monto": amount,
                "registradoPor": admin_id
            }
            
            loan["pagosRealizados"].append(payment)
            
            # Verificar si se completó el pago
            if loan["saldoPendiente"] <= 0:
                loan["estado"] = "pagado"
                loan["fechaFinalizacion"] = timestamp
                
            # Actualizar estadísticas del inversionista
            investor_id = loan["inversionista"]
            if investor_id in self.investors:
                investor = self.investors[investor_id]
                investor["performance_stats"]["active_loans"] -= Decimal(str(amount))
                investor["updated_at"] = timestamp
                
                # Registrar transacción
                transaction = await self._register_transaction(
                    investor_id=investor_id,
                    transaction_type=TransactionType.LOAN_PAYMENT,
                    amount=-amount,  # Negativo porque es un pago
                    description=f"Pago de préstamo {loan_id}",
                    related_id=None,
                    admin_id=admin_id,
                    metadata={"loan_id": loan_id}
                )
                
                # Guardar cambios
                if self.db:
                    await self.db.set(f"investor:{investor_id}", json.dumps(investor))
            else:
                self.logger.warning(f"Inversionista {investor_id} no encontrado al procesar pago")
                transaction = None
                
            # Guardar cambios del préstamo
            if self.db:
                await self.db.set(f"loan:{loan_id}", json.dumps(loan))
                
            self.logger.info(f"Pago de préstamo procesado: ${amount} para préstamo {loan_id}")
            
            return {
                "success": True,
                "loan": loan,
                "payment": payment,
                "transaction": transaction,
                "remaining_balance": loan["saldoPendiente"],
                "status": loan["estado"]
            }
            
        except Exception as e:
            self.logger.error(f"Error al procesar pago de préstamo: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def create_referral_contract(self,
                                      admin_id: str,
                                      investor_id: str,
                                      percentage: float,
                                      duration_months: int = 6,
                                      super_admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Crear contrato de comisión entre admin e inversionista.
        
        Args:
            admin_id: ID del administrador que recibirá la comisión
            investor_id: ID del inversionista que pagará la comisión
            percentage: Porcentaje de comisión (0-100)
            duration_months: Duración del contrato en meses
            super_admin_id: ID del superadmin que autoriza
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si las comisiones están permitidas
            if not self.daily_stats["commissions_allowed"]:
                return {"success": False, "error": "Las comisiones están deshabilitadas temporalmente"}
                
            # Validar parámetros
            if percentage <= 0 or percentage > 50:
                return {"success": False, "error": "El porcentaje debe estar entre 0.1% y 50%"}
                
            if duration_months <= 0 or duration_months > 24:
                return {"success": False, "error": "La duración debe estar entre 1 y 24 meses"}
                
            # Verificar inversionista
            if investor_id not in self.investors:
                return {"success": False, "error": "Inversionista no encontrado"}
                
            investor = self.investors[investor_id]
            
            # Verificar estado activo
            if investor["status"] != InvestorStatus.ACTIVE:
                return {"success": False, "error": "El inversionista no está activo"}
                
            # Verificar si ya existe un contrato activo
            existing_contracts = [
                contract for contract in self.referral_contracts.values()
                if contract["inversionista"] == investor_id 
                and contract["admin"] == admin_id
                and contract["estado"] == "activo"
            ]
            
            if existing_contracts:
                return {"success": False, "error": "Ya existe un contrato activo entre este admin e inversionista"}
                
            # Crear contrato (inicialmente pendiente)
            contract_id = generate_id()
            timestamp = datetime.now().isoformat()
            end_date = (datetime.now() + timedelta(days=30*duration_months)).isoformat()
            
            contract = {
                "id": contract_id,
                "admin": admin_id,
                "inversionista": investor_id,
                "porcentaje": percentage,
                "estado": "pendiente",
                "fechaInicio": timestamp,
                "fechaFin": end_date,
                "autorizadoPor": super_admin_id,
                "fechaAceptacion": None,
                "comisionesAcumuladas": 0
            }
            
            # Registrar contrato
            self.referral_contracts[contract_id] = contract
            
            # Guardar en base de datos
            if self.db:
                await self.db.set(f"referral_contract:{contract_id}", json.dumps(contract))
                
            self.logger.info(f"Contrato de comisión creado: {percentage}% entre admin {admin_id} e inversionista {investor_id}")
            
            return {
                "success": True,
                "contract": contract,
                "message": "Contrato creado y pendiente de aceptación por el inversionista"
            }
            
        except Exception as e:
            self.logger.error(f"Error al crear contrato de comisión: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def accept_referral_contract(self,
                                      contract_id: str,
                                      investor_id: str) -> Dict[str, Any]:
        """
        Aceptar contrato de comisión por parte del inversionista.
        
        Args:
            contract_id: ID del contrato
            investor_id: ID del inversionista que acepta
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar contrato
            if contract_id not in self.referral_contracts:
                return {"success": False, "error": "Contrato no encontrado"}
                
            contract = self.referral_contracts[contract_id]
            
            # Verificar estado
            if contract["estado"] != "pendiente":
                return {"success": False, "error": "El contrato no está en estado pendiente"}
                
            # Verificar inversionista
            if contract["inversionista"] != investor_id:
                return {"success": False, "error": "Este contrato no pertenece al inversionista indicado"}
                
            # Actualizar contrato
            timestamp = datetime.now().isoformat()
            contract["estado"] = "activo"
            contract["fechaAceptacion"] = timestamp
            
            # Guardar en base de datos
            if self.db:
                await self.db.set(f"referral_contract:{contract_id}", json.dumps(contract))
                
            self.logger.info(f"Contrato de comisión aceptado: {contract_id} por inversionista {investor_id}")
            
            return {
                "success": True,
                "contract": contract,
                "message": "Contrato aceptado correctamente"
            }
            
        except Exception as e:
            self.logger.error(f"Error al aceptar contrato de comisión: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def update_daily_stats(self,
                                bonus_enabled: Optional[bool] = None,
                                loans_enabled: Optional[bool] = None,
                                commissions_allowed: Optional[bool] = None,
                                transfer_limit: Optional[float] = None,
                                notes: Optional[str] = None,
                                admin_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Actualizar estadísticas diarias del sistema.
        
        Args:
            bonus_enabled: Activar/desactivar bonos de rendimiento
            loans_enabled: Activar/desactivar préstamos
            commissions_allowed: Permitir/prohibir comisiones
            transfer_limit: Límite de transferencia diaria
            notes: Notas del día
            admin_id: ID del administrador que realiza el cambio
            
        Returns:
            Resultado de la operación
        """
        try:
            modified = False
            
            # Actualizar campos si se proporcionan
            if bonus_enabled is not None and self.daily_stats["bonus_enabled"] != bonus_enabled:
                self.daily_stats["bonus_enabled"] = bonus_enabled
                modified = True
                
            if loans_enabled is not None and self.daily_stats["loans_enabled"] != loans_enabled:
                self.daily_stats["loans_enabled"] = loans_enabled
                modified = True
                
            if commissions_allowed is not None and self.daily_stats["commissions_allowed"] != commissions_allowed:
                self.daily_stats["commissions_allowed"] = commissions_allowed
                modified = True
                
            if transfer_limit is not None:
                self.daily_stats["transfer_limit"] = Decimal(str(transfer_limit))
                modified = True
                
            if notes is not None:
                self.daily_stats["notes"] = notes
                modified = True
                
            if modified:
                # Guardar en base de datos
                if self.db:
                    # Convertir Decimal a float para JSON
                    stats_copy = self.daily_stats.copy()
                    stats_copy["transfer_limit"] = float(stats_copy["transfer_limit"])
                    
                    await self.db.set("system:daily_stats", json.dumps(stats_copy))
                    
                self.logger.info(f"Estadísticas diarias actualizadas por {admin_id}")
                
                return {
                    "success": True,
                    "daily_stats": self.daily_stats,
                    "message": "Estadísticas diarias actualizadas correctamente"
                }
            else:
                return {
                    "success": True,
                    "message": "No se realizaron cambios"
                }
                
        except Exception as e:
            self.logger.error(f"Error al actualizar estadísticas diarias: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_transactions(self,
                              investor_id: Optional[str] = None,
                              transaction_type: Optional[str] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              limit: int = 100) -> Dict[str, Any]:
        """
        Obtener transacciones filtradas.
        
        Args:
            investor_id: Filtrar por inversionista (opcional)
            transaction_type: Filtrar por tipo (opcional)
            start_date: Fecha inicio formato ISO (opcional)
            end_date: Fecha fin formato ISO (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de transacciones
        """
        try:
            # Aplicar filtros
            filtered_transactions = []
            
            for transaction in self.transactions.values():
                # Filtrar por inversionista
                if investor_id and transaction["investor_id"] != investor_id:
                    continue
                    
                # Filtrar por tipo
                if transaction_type and transaction["type"] != transaction_type:
                    continue
                    
                # Filtrar por fecha inicio
                if start_date and transaction["timestamp"] < start_date:
                    continue
                    
                # Filtrar por fecha fin
                if end_date and transaction["timestamp"] > end_date:
                    continue
                    
                filtered_transactions.append(transaction)
            
            # Ordenar por fecha (más reciente primero)
            filtered_transactions.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Aplicar límite
            result_transactions = filtered_transactions[:limit]
            
            return {
                "success": True,
                "transactions": result_transactions,
                "total": len(filtered_transactions),
                "shown": len(result_transactions)
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener transacciones: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_active_loans(self, investor_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener préstamos activos.
        
        Args:
            investor_id: Filtrar por inversionista (opcional)
            
        Returns:
            Lista de préstamos activos
        """
        try:
            # Filtrar préstamos
            if investor_id:
                loans = [
                    loan for loan in self.active_loans.values()
                    if loan["inversionista"] == investor_id and loan["estado"] == "activo"
                ]
            else:
                loans = [
                    loan for loan in self.active_loans.values()
                    if loan["estado"] == "activo"
                ]
                
            # Ordenar por fecha (más reciente primero)
            loans.sort(key=lambda x: x["fechaInicio"], reverse=True)
            
            return {
                "success": True,
                "loans": loans,
                "count": len(loans)
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener préstamos activos: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_referral_contracts(self, 
                                    investor_id: Optional[str] = None,
                                    admin_id: Optional[str] = None,
                                    status: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener contratos de comisión filtrados.
        
        Args:
            investor_id: Filtrar por inversionista (opcional)
            admin_id: Filtrar por administrador (opcional)
            status: Filtrar por estado (opcional)
            
        Returns:
            Lista de contratos
        """
        try:
            # Aplicar filtros
            filtered_contracts = []
            
            for contract in self.referral_contracts.values():
                # Filtrar por inversionista
                if investor_id and contract["inversionista"] != investor_id:
                    continue
                    
                # Filtrar por admin
                if admin_id and contract["admin"] != admin_id:
                    continue
                    
                # Filtrar por estado
                if status and contract["estado"] != status:
                    continue
                    
                filtered_contracts.append(contract)
            
            # Ordenar por fecha (más reciente primero)
            filtered_contracts.sort(key=lambda x: x["fechaInicio"], reverse=True)
            
            return {
                "success": True,
                "contracts": filtered_contracts,
                "count": len(filtered_contracts)
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener contratos de comisión: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de rendimiento generales.
        
        Returns:
            Estadísticas generales
        """
        try:
            # Calcular estadísticas generales
            active_investors = len([i for i in self.investors.values() if i["status"] == InvestorStatus.ACTIVE])
            total_capital = float(self.total_capital)
            total_invested = sum(float(i["invested_capital"]) for i in self.investors.values())
            total_profit_distributed = sum(float(i["total_profit"]) for i in self.investors.values())
            active_loans_count = len([l for l in self.active_loans.values() if l["estado"] == "activo"])
            active_loans_amount = sum(l["saldoPendiente"] for l in self.active_loans.values() if l["estado"] == "activo")
            
            # Calcular bonos de rendimiento
            total_bonus = sum(float(i["performance_stats"]["total_bonus"]) for i in self.investors.values())
            
            stats = {
                "investor_count": {
                    "total": len(self.investors),
                    "active": active_investors
                },
                "capital": {
                    "total": total_capital,
                    "invested": total_invested,
                    "profit_distributed": total_profit_distributed
                },
                "loans": {
                    "active_count": active_loans_count,
                    "active_amount": active_loans_amount
                },
                "bonuses": {
                    "total_amount": total_bonus
                },
                "system_status": self.daily_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "stats": stats
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener estadísticas de rendimiento: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _apply_performance_bonuses(self) -> None:
        """Aplicar bonos de rendimiento si el sistema lo permite."""
        try:
            # Verificar si los bonos están habilitados
            if not self.daily_stats["bonus_enabled"]:
                return
                
            # Verificar capital total necesario para bonos
            if self.total_capital < 5000:
                return
                
            # Determinar porcentaje de bono
            bonus_percentage = 0.05  # 5% por defecto
            
            if self.total_capital >= 10000:
                bonus_percentage = 0.07  # 7%
                
            # Aplicar bonos a inversionistas activos
            active_investors = {
                investor_id: investor
                for investor_id, investor in self.investors.items()
                if investor["status"] == InvestorStatus.ACTIVE
            }
            
            if not active_investors:
                return
                
            self.logger.info(f"Aplicando bonos de rendimiento ({bonus_percentage*100}%) a {len(active_investors)} inversionistas")
            
            timestamp = datetime.now().isoformat()
            
            for investor_id, investor in active_investors.items():
                # Calcular bono
                balance = float(investor["balance"])
                bonus_amount = balance * bonus_percentage
                
                # Actualizar saldo
                investor["balance"] += Decimal(str(bonus_amount))
                investor["total_profit"] += Decimal(str(bonus_amount))
                investor["performance_stats"]["total_bonus"] += Decimal(str(bonus_amount))
                investor["updated_at"] = timestamp
                
                # Registrar bono
                bonus_id = generate_id()
                
                bonus = {
                    "id": bonus_id,
                    "inversionista": investor_id,
                    "montoGanancia": balance,
                    "bonoAplicado": bonus_amount,
                    "fecha": timestamp,
                    "aprobadoPor": "system",
                    "nivel": f"{bonus_percentage*100}%"
                }
                
                self.performance_bonuses[bonus_id] = bonus
                
                # Registrar transacción
                await self._register_transaction(
                    investor_id=investor_id,
                    transaction_type=TransactionType.BONUS,
                    amount=bonus_amount,
                    description=f"Bono por rendimiento ({bonus_percentage*100}%)",
                    related_id=None,
                    admin_id=None,
                    metadata={"bonus_id": bonus_id}
                )
                
                # Guardar en base de datos
                if self.db:
                    await self.db.set(f"investor:{investor_id}", json.dumps(investor))
                    await self.db.set(f"performance_bonus:{bonus_id}", json.dumps(bonus))
                    
            self.logger.info(f"Bonos de rendimiento aplicados a {len(active_investors)} inversionistas")
            
        except Exception as e:
            self.logger.error(f"Error al aplicar bonos de rendimiento: {str(e)}")
    
    async def _register_transaction(self,
                                   investor_id: str,
                                   transaction_type: str,
                                   amount: float,
                                   description: str,
                                   related_id: Optional[str],
                                   admin_id: Optional[str],
                                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registrar transacción en el sistema.
        
        Args:
            investor_id: ID del inversionista
            transaction_type: Tipo de transacción
            amount: Cantidad (positiva o negativa)
            description: Descripción
            related_id: ID relacionado (opcional)
            admin_id: ID del administrador (opcional)
            metadata: Metadatos adicionales (opcional)
            
        Returns:
            Transacción registrada
        """
        try:
            transaction_id = generate_id()
            timestamp = datetime.now().isoformat()
            
            transaction = {
                "id": transaction_id,
                "investor_id": investor_id,
                "type": transaction_type,
                "amount": amount,
                "description": description,
                "timestamp": timestamp,
                "related_id": related_id,
                "admin_id": admin_id,
                "metadata": metadata or {}
            }
            
            # Guardar en memoria
            self.transactions[transaction_id] = transaction
            
            # Guardar en base de datos
            if self.db:
                await self.db.set(f"transaction:{transaction_id}", json.dumps(transaction))
                
            return transaction
            
        except Exception as e:
            self.logger.error(f"Error al registrar transacción: {str(e)}")
            return {}
    
    def _clean_for_response(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convertir objetos Decimal a float para respuestas JSON.
        
        Args:
            obj: Objeto a limpiar
            
        Returns:
            Objeto limpio para respuesta
        """
        result = {}
        
        for key, value in obj.items():
            if isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, dict):
                result[key] = self._clean_for_response(value)
            else:
                result[key] = value
                
        return result