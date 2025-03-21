#!/usr/bin/env python
"""
Script para inicializar las tablas de Paper Trading en la base de datos.

Este script crea las tablas necesarias para el modo de Paper Trading
y opcionalmente inicializa cuentas de prueba para usar el sistema.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Añadimos el directorio raíz al path para importar los módulos de Genesis
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.db.models import Base
from genesis.db.repository import Repository
from genesis.db.paper_trading_models import (
    PaperTradingAccount, PaperAssetBalance, 
    PaperOrder, PaperTrade, PaperBalanceSnapshot
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear ejemplo de cuenta paper trading con 10,000 USD
CREATE_EXAMPLE_ACCOUNT = True

async def initialize_database():
    """Inicializar la base de datos para paper trading."""
    logger.info("Inicializando base de datos para Paper Trading...")
    
    repo = Repository()
    
    try:
        # Crear tablas si no existen
        await repo.create_tables(Base)
        logger.info("Tablas creadas o ya existentes")
        
        # Crear cuenta de ejemplo si se solicitó
        if CREATE_EXAMPLE_ACCOUNT:
            # Verificar si ya existe una cuenta de ejemplo
            existing_accounts = await repo.query(
                PaperTradingAccount,
                "name = 'Cuenta de Ejemplo'"
            )
            
            if existing_accounts:
                logger.info(f"La cuenta de ejemplo ya existe (ID: {existing_accounts[0].id})")
            else:
                # Crear una nueva cuenta
                account = PaperTradingAccount(
                    name="Cuenta de Ejemplo",
                    description="Cuenta de trading simulado para pruebas",
                    initial_balance_usd=10000.0,
                    current_balance_usd=10000.0,
                    config={
                        "fee_rate": 0.001,  # 0.1% fee (similar a Binance)
                        "slippage": 0.0005  # 0.05% slippage
                    }
                )
                
                account_id = await repo.create(account)
                
                # Agregar balance inicial en USDT
                balance = PaperAssetBalance(
                    account_id=account_id,
                    asset="USDT",
                    total=10000.0,
                    available=10000.0,
                    locked=0.0
                )
                
                await repo.create(balance)
                
                logger.info(f"Creada cuenta de ejemplo con {balance.total} USDT (ID: {account_id})")
                
        logger.info("Inicialización completada correctamente")
        
    except Exception as e:
        logger.error(f"Error al inicializar base de datos: {e}")
        raise

async def main():
    """Función principal del script."""
    try:
        await initialize_database()
    except Exception as e:
        logger.error(f"Error en la inicialización: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())