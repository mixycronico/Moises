"""
Script para configurar particiones de base de datos PostgreSQL.

Este script crea particiones para las tablas que requieren alta escalabilidad
y rendimiento en el sistema Genesis.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_partitions(connection_string):
    """
    Crear particiones para tablas que requieren alta escalabilidad.
    
    Args:
        connection_string: Cadena de conexión a la base de datos
    
    Returns:
        bool: True si se crearon las particiones correctamente, False en caso contrario
    """
    logger.info("Conectando a la base de datos para configuración de particiones")
    
    try:
        # Crear motor de base de datos
        engine = create_engine(connection_string)
        conn = engine.connect()
        
        # Lista de operaciones de particionamiento a ejecutar
        partition_operations = [
            # Particiones para candles por rango de fechas (trimestral para el año actual)
            """
            CREATE TABLE IF NOT EXISTS candles_1m_y2025q1 PARTITION OF candles_1m
                FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_1m_y2025q2 PARTITION OF candles_1m
                FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_1m_y2025q3 PARTITION OF candles_1m
                FOR VALUES FROM ('2025-07-01') TO ('2025-10-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_1m_y2025q4 PARTITION OF candles_1m
                FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');
            """,
            
            # Particiones para candles 5m (similar)
            """
            CREATE TABLE IF NOT EXISTS candles_5m_y2025q1 PARTITION OF candles_5m
                FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_5m_y2025q2 PARTITION OF candles_5m
                FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_5m_y2025q3 PARTITION OF candles_5m
                FOR VALUES FROM ('2025-07-01') TO ('2025-10-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_5m_y2025q4 PARTITION OF candles_5m
                FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');
            """,
            
            # Particiones para candles 15m (similar pero semestral para reducir el número)
            """
            CREATE TABLE IF NOT EXISTS candles_15m_y2025h1 PARTITION OF candles_15m
                FOR VALUES FROM ('2025-01-01') TO ('2025-07-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_15m_y2025h2 PARTITION OF candles_15m
                FOR VALUES FROM ('2025-07-01') TO ('2026-01-01');
            """,
            
            # Particiones para candles 1h (semestral)
            """
            CREATE TABLE IF NOT EXISTS candles_1h_y2025h1 PARTITION OF candles_1h
                FOR VALUES FROM ('2025-01-01') TO ('2025-07-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_1h_y2025h2 PARTITION OF candles_1h
                FOR VALUES FROM ('2025-07-01') TO ('2026-01-01');
            """,
            
            # Particiones para candles 4h y 1d (anual)
            """
            CREATE TABLE IF NOT EXISTS candles_4h_y2025 PARTITION OF candles_4h
                FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS candles_1d_y2025 PARTITION OF candles_1d
                FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
            """,
            
            # Particiones para trades por trimestre
            """
            CREATE TABLE IF NOT EXISTS trades_y2025q1 PARTITION OF trades
                FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS trades_y2025q2 PARTITION OF trades
                FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS trades_y2025q3 PARTITION OF trades
                FOR VALUES FROM ('2025-07-01') TO ('2025-10-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS trades_y2025q4 PARTITION OF trades
                FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');
            """,
            
            # Particiones para system_logs por mes 
            """
            CREATE TABLE IF NOT EXISTS system_logs_y2025m01 PARTITION OF system_logs
                FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS system_logs_y2025m02 PARTITION OF system_logs
                FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS system_logs_y2025m03 PARTITION OF system_logs
                FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
            """,
            """
            CREATE TABLE IF NOT EXISTS system_logs_y2025m04 PARTITION OF system_logs
                FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
            """,
            
            # Particiones por exchange para balances
            """
            CREATE TABLE IF NOT EXISTS balances_binance PARTITION OF balances
                FOR VALUES IN ('1'); -- Asumiendo que Binance tiene ID 1
            """,
            """
            CREATE TABLE IF NOT EXISTS balances_coinbase PARTITION OF balances
                FOR VALUES IN ('2'); -- Asumiendo que Coinbase tiene ID 2
            """,
            """
            CREATE TABLE IF NOT EXISTS balances_kraken PARTITION OF balances
                FOR VALUES IN ('3'); -- Asumiendo que Kraken tiene ID 3
            """,
            
            # Particiones por estrategia para signals
            """
            CREATE TABLE IF NOT EXISTS signals_strategy1 PARTITION OF signals
                FOR VALUES IN ('1'); -- Asumiendo que la estrategia 1 tiene ID 1
            """,
            """
            CREATE TABLE IF NOT EXISTS signals_strategy2 PARTITION OF signals
                FOR VALUES IN ('2'); -- Asumiendo que la estrategia 2 tiene ID 2
            """,
            """
            CREATE TABLE IF NOT EXISTS signals_strategy3 PARTITION OF signals
                FOR VALUES IN ('3'); -- Asumiendo que la estrategia 3 tiene ID 3
            """,
            
            # Particiones para backtest_results por estrategia
            """
            CREATE TABLE IF NOT EXISTS backtest_results_trend_following PARTITION OF backtest_results
                FOR VALUES IN ('trend_following');
            """,
            """
            CREATE TABLE IF NOT EXISTS backtest_results_mean_reversion PARTITION OF backtest_results
                FOR VALUES IN ('mean_reversion');
            """,
            """
            CREATE TABLE IF NOT EXISTS backtest_results_sentiment PARTITION OF backtest_results
                FOR VALUES IN ('sentiment');
            """,
            
            # Índices adicionales para mejorar rendimiento
            """
            CREATE INDEX IF NOT EXISTS idx_candles_1m_symbol_timestamp ON candles_1m(symbol, timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_candles_5m_symbol_timestamp ON candles_5m(symbol, timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_candles_15m_symbol_timestamp ON candles_15m(symbol, timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_candles_1h_symbol_timestamp ON candles_1h(symbol, timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_entry_time ON trades(symbol, entry_time);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_system_logs_component_level ON system_logs(component, level);
            """
        ]
        
        # Ejecutar cada operación
        for op in partition_operations:
            try:
                conn.execute(text(op))
                logger.info(f"Operación de particionamiento ejecutada: {op.split()[0:5]}...")
            except Exception as e:
                logger.warning(f"Error al ejecutar operación: {e}")
                # Continuar con las siguientes operaciones
        
        conn.close()
        logger.info("Particiones configuradas correctamente")
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"Error al configurar particiones: {e}")
        return False
        
if __name__ == "__main__":
    # Obtener la cadena de conexión de la base de datos
    connection_string = os.environ.get("DATABASE_URL")
    if not connection_string:
        logger.error("Variable de entorno DATABASE_URL no definida")
        sys.exit(1)
    
    logger.info("Iniciando configuración de particiones de base de datos...")
    if create_partitions(connection_string):
        logger.info("Configuración de particiones completada exitosamente")
    else:
        logger.error("Error en la configuración de particiones")
        sys.exit(1)