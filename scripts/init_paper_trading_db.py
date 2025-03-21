#!/usr/bin/env python
"""
Script para inicializar las tablas de Paper Trading en la base de datos.

Este script crea las tablas necesarias para el modo de Paper Trading
y opcionalmente inicializa cuentas de prueba para usar el sistema.
"""

import os
import sys
import logging
import asyncio
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("init_paper_trading_db")

# Asegurarse de que podemos importar los módulos de Genesis
sys.path.insert(0, os.getcwd())

# Importar modelos
from genesis.db.paper_trading_models import Base, PaperTradingAccount, PaperTradingBalance, MarketData

# Directorios y constantes
DATA_DIR = './data/testnet'
DB_URL = os.environ.get('DATABASE_URL')

async def initialize_database():
    """Inicializar la base de datos para paper trading."""
    if not DB_URL:
        logger.error("No se encontró la variable de entorno DATABASE_URL")
        return False
    
    logger.info(f"Conectando a la base de datos: {DB_URL}")
    
    try:
        # Crear el motor de la base de datos y la sesión
        engine = create_engine(DB_URL)
        
        # Crear tablas
        logger.info("Creando tablas de paper trading...")
        Base.metadata.create_all(engine)
        logger.info("Tablas creadas correctamente.")
        
        # Crear sesión
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Inicializar cuenta de prueba
        await create_test_account(session)
        
        # Importar datos históricos
        await import_historical_data(session)
        
        session.close()
        return True
    
    except Exception as e:
        logger.error(f"Error al inicializar la base de datos: {e}")
        return False

async def create_test_account(session):
    """Crear una cuenta de prueba para paper trading."""
    try:
        # Verificar si ya existe una cuenta de prueba
        existing_account = session.query(PaperTradingAccount).filter_by(name="Test Account").first()
        
        if existing_account:
            logger.info(f"Cuenta de prueba ya existe (ID: {existing_account.id})")
            return existing_account
        
        # Crear nueva cuenta
        logger.info("Creando cuenta de prueba para paper trading...")
        account = PaperTradingAccount(
            name="Test Account",
            description="Cuenta automática para pruebas de paper trading",
            user_id=1,  # Usuario de prueba
            initial_balance_usd=10000.0,
            is_active=True
        )
        session.add(account)
        session.flush()  # Para obtener el ID asignado
        
        # Crear saldos iniciales
        balances = [
            PaperTradingBalance(account_id=account.id, asset="USDT", free=10000.0),
            PaperTradingBalance(account_id=account.id, asset="BTC", free=0.0),
            PaperTradingBalance(account_id=account.id, asset="ETH", free=0.0),
            PaperTradingBalance(account_id=account.id, asset="BNB", free=0.0)
        ]
        
        for balance in balances:
            session.add(balance)
        
        session.commit()
        logger.info(f"Cuenta de prueba creada con ID: {account.id}")
        return account
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error al crear cuenta de prueba: {e}")
        return None

async def import_historical_data(session):
    """Importar datos históricos desde archivos CSV a la base de datos."""
    try:
        data_dir = Path(DATA_DIR)
        if not data_dir.exists():
            logger.warning(f"Directorio de datos {DATA_DIR} no encontrado")
            return False
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No se encontraron archivos CSV en {DATA_DIR}")
            return False
        
        logger.info(f"Encontrados {len(csv_files)} archivos CSV para importar")
        
        # Eliminar datos existentes para evitar duplicados
        logger.info("Eliminando datos existentes...")
        session.execute(text("DELETE FROM market_data WHERE source = 'testnet'"))
        session.commit()
        
        total_records = 0
        
        for csv_file in csv_files:
            try:
                # Extraer símbolo y timeframe del nombre del archivo
                filename = csv_file.stem  # Nombre sin extensión
                parts = filename.split("_")
                
                if len(parts) >= 3:
                    symbol = f"{parts[0]}/{parts[1]}"
                    timeframe = parts[2]
                else:
                    logger.warning(f"Formato de nombre de archivo no reconocido: {filename}")
                    continue
                
                logger.info(f"Importando datos para {symbol} ({timeframe}) desde {csv_file}")
                
                # Leer CSV
                df = pd.read_csv(csv_file)
                record_count = 0
                
                # Insertar datos en bloques para mejor rendimiento
                batch_size = 100
                data_rows = []
                
                for index, row in df.iterrows():
                    # Convertir timestamp a datetime si es necesario
                    if 'datetime' in row:
                        timestamp = datetime.fromisoformat(row['datetime'].replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.fromtimestamp(row['timestamp'] / 1000)  # Convertir de milisegundos a segundos
                    
                    market_data = MarketData(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        source='testnet'
                    )
                    data_rows.append(market_data)
                    record_count += 1
                    
                    # Insertar en lotes
                    if len(data_rows) >= batch_size:
                        session.bulk_save_objects(data_rows)
                        session.commit()
                        data_rows = []
                
                # Insertar cualquier registro restante
                if data_rows:
                    session.bulk_save_objects(data_rows)
                    session.commit()
                
                logger.info(f"Importados {record_count} registros para {symbol} ({timeframe})")
                total_records += record_count
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error al importar archivo {csv_file}: {e}")
        
        logger.info(f"Importación completada. Total de registros importados: {total_records}")
        return True
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error en la importación de datos históricos: {e}")
        return False

async def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Inicializar base de datos de Paper Trading')
    parser.add_argument('--reset', action='store_true', help='Eliminar y recrear todas las tablas')
    args = parser.parse_args()
    
    if args.reset:
        logger.warning("Modo RESET activado - se eliminarán todas las tablas existentes")
        # Implementar lógica para eliminar tablas si es necesario
    
    success = await initialize_database()
    
    if success:
        logger.info("Base de datos de Paper Trading inicializada correctamente")
    else:
        logger.error("Falló la inicialización de la base de datos de Paper Trading")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Iniciando script de inicialización de BD para Paper Trading")
    asyncio.run(main())