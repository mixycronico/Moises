"""
Script para inicializar las tablas de diccionario del sistema Genesis.

Este script crea los registros iniciales para las tablas de diccionario
que serán utilizadas por el sistema de trading.
"""

import asyncio
import logging
import sys
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append('.')
from genesis.db.models import (
    Base, DictSignalType, DictTradeStatus, DictStrategyType,
    DictTradingPair, DictRiskLevel
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Usar la variable de entorno DATABASE_URL (requerido en producción)
from os import environ
DATABASE_URL = environ.get('DATABASE_URL', 'sqlite:///genesis.db')

def init_signal_types(session):
    """Inicializar tipos de señales de trading."""
    logger.info("Inicializando tipos de señales...")
    
    signal_types = [
        {
            'code': 'BUY',
            'name': 'Comprar',
            'description': 'Señal para entrar en una posición larga',
            'color': '#4CAF50',  # Verde
            'icon': 'trending_up'
        },
        {
            'code': 'SELL',
            'name': 'Vender',
            'description': 'Señal para entrar en una posición corta',
            'color': '#F44336',  # Rojo
            'icon': 'trending_down'
        },
        {
            'code': 'HOLD',
            'name': 'Mantener',
            'description': 'Señal para mantener posición actual',
            'color': '#03A9F4',  # Azul claro
            'icon': 'trending_flat'
        },
        {
            'code': 'EXIT',
            'name': 'Salir',
            'description': 'Señal para cerrar posición actual',
            'color': '#FF9800',  # Naranja
            'icon': 'exit_to_app'
        },
        {
            'code': 'CLOSE',
            'name': 'Cerrar Todo',
            'description': 'Señal para cerrar todas las posiciones',
            'color': '#9C27B0',  # Púrpura
            'icon': 'close'
        }
    ]
    
    for data in signal_types:
        # Verificar si ya existe
        existing = session.query(DictSignalType).filter_by(code=data['code']).first()
        if existing:
            logger.info(f"Señal {data['code']} ya existe, actualizando...")
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            logger.info(f"Creando nueva señal: {data['code']}")
            signal = DictSignalType(
                code=data['code'],
                name=data['name'],
                description=data['description'],
                color=data['color'],
                icon=data['icon'],
                is_active=True,
                created_at=datetime.now()
            )
            session.add(signal)
    
    session.commit()
    logger.info("Tipos de señales inicializados correctamente.")

def init_trade_status(session):
    """Inicializar estados de operaciones."""
    logger.info("Inicializando estados de operaciones...")
    
    trade_statuses = [
        {
            'code': 'OPEN',
            'name': 'Abierta',
            'description': 'Operación actualmente abierta',
            'color': '#4CAF50',  # Verde
            'icon': 'radio_button_checked'
        },
        {
            'code': 'CLOSED',
            'name': 'Cerrada',
            'description': 'Operación completada y cerrada',
            'color': '#03A9F4',  # Azul claro
            'icon': 'check_circle'
        },
        {
            'code': 'CANCELED',
            'name': 'Cancelada',
            'description': 'Operación cancelada antes de ejecutarse',
            'color': '#F44336',  # Rojo
            'icon': 'cancel'
        },
        {
            'code': 'PENDING',
            'name': 'Pendiente',
            'description': 'Operación pendiente de ejecución',
            'color': '#FF9800',  # Naranja
            'icon': 'pending'
        },
        {
            'code': 'PARTIALLY_FILLED',
            'name': 'Parcialmente Ejecutada',
            'description': 'Operación parcialmente ejecutada',
            'color': '#9C27B0',  # Púrpura
            'icon': 'incomplete_circle'
        }
    ]
    
    for data in trade_statuses:
        # Verificar si ya existe
        existing = session.query(DictTradeStatus).filter_by(code=data['code']).first()
        if existing:
            logger.info(f"Estado {data['code']} ya existe, actualizando...")
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            logger.info(f"Creando nuevo estado: {data['code']}")
            status = DictTradeStatus(
                code=data['code'],
                name=data['name'],
                description=data['description'],
                color=data['color'],
                icon=data['icon'],
                is_active=True,
                created_at=datetime.now()
            )
            session.add(status)
    
    session.commit()
    logger.info("Estados de operaciones inicializados correctamente.")

def init_strategy_types(session):
    """Inicializar tipos de estrategias."""
    logger.info("Inicializando tipos de estrategias...")
    
    strategy_types = [
        {
            'code': 'RSI',
            'name': 'Relative Strength Index',
            'description': 'Estrategia basada en el índice de fuerza relativa',
            'category': 'TECHNICAL',
            'complexity': 'BASIC',
            'risk_level': 'MEDIUM',
            'params_schema': {
                'period': {'type': 'integer', 'default': 14, 'min': 2, 'max': 100},
                'overbought': {'type': 'integer', 'default': 70, 'min': 50, 'max': 90},
                'oversold': {'type': 'integer', 'default': 30, 'min': 10, 'max': 50}
            }
        },
        {
            'code': 'BOLLINGER',
            'name': 'Bandas de Bollinger',
            'description': 'Estrategia basada en las bandas de Bollinger',
            'category': 'TECHNICAL',
            'complexity': 'INTERMEDIATE',
            'risk_level': 'MEDIUM',
            'params_schema': {
                'period': {'type': 'integer', 'default': 20, 'min': 5, 'max': 100},
                'std_dev': {'type': 'number', 'default': 2.0, 'min': 0.5, 'max': 4.0}
            }
        },
        {
            'code': 'MA_CROSSOVER',
            'name': 'Cruce de Medias Móviles',
            'description': 'Estrategia basada en el cruce de dos medias móviles',
            'category': 'TECHNICAL',
            'complexity': 'BASIC',
            'risk_level': 'MEDIUM',
            'params_schema': {
                'fast_period': {'type': 'integer', 'default': 9, 'min': 2, 'max': 50},
                'slow_period': {'type': 'integer', 'default': 21, 'min': 5, 'max': 200}
            }
        },
        {
            'code': 'MACD',
            'name': 'MACD',
            'description': 'Estrategia basada en el indicador de convergencia/divergencia de medias móviles',
            'category': 'TECHNICAL',
            'complexity': 'INTERMEDIATE',
            'risk_level': 'MEDIUM',
            'params_schema': {
                'fast_period': {'type': 'integer', 'default': 12, 'min': 2, 'max': 50},
                'slow_period': {'type': 'integer', 'default': 26, 'min': 5, 'max': 200},
                'signal_period': {'type': 'integer', 'default': 9, 'min': 2, 'max': 50}
            }
        },
        {
            'code': 'SENTIMENT',
            'name': 'Análisis de Sentimiento',
            'description': 'Estrategia basada en análisis de sentimiento de mercado',
            'category': 'FUNDAMENTAL',
            'complexity': 'ADVANCED',
            'risk_level': 'HIGH',
            'params_schema': {
                'sentiment_threshold': {'type': 'number', 'default': 0.7, 'min': 0.1, 'max': 0.9},
                'twitter_weight': {'type': 'number', 'default': 0.5, 'min': 0.0, 'max': 1.0},
                'news_weight': {'type': 'number', 'default': 0.5, 'min': 0.0, 'max': 1.0}
            }
        }
    ]
    
    for data in strategy_types:
        # Verificar si ya existe
        existing = session.query(DictStrategyType).filter_by(code=data['code']).first()
        if existing:
            logger.info(f"Estrategia {data['code']} ya existe, actualizando...")
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            logger.info(f"Creando nueva estrategia: {data['code']}")
            strategy = DictStrategyType(
                code=data['code'],
                name=data['name'],
                description=data['description'],
                category=data['category'],
                complexity=data['complexity'],
                risk_level=data['risk_level'],
                params_schema=data['params_schema'],
                is_active=True,
                created_at=datetime.now()
            )
            session.add(strategy)
    
    session.commit()
    logger.info("Tipos de estrategias inicializados correctamente.")

def init_trading_pairs(session):
    """Inicializar pares de trading."""
    logger.info("Inicializando pares de trading...")
    
    trading_pairs = [
        {
            'symbol': 'BTC/USDT',
            'base_asset': 'BTC',
            'quote_asset': 'USDT',
            'min_order_size': 0.001,
            'max_order_size': 1000.0,
            'price_precision': 2,
            'quantity_precision': 6,
            'additional_data': {
                'popular': True,
                'category': 'MAJOR',
                'description': 'Bitcoin vs Tether',
                'is_stablecoin_pair': True
            }
        },
        {
            'symbol': 'ETH/USDT',
            'base_asset': 'ETH',
            'quote_asset': 'USDT',
            'min_order_size': 0.01,
            'max_order_size': 5000.0,
            'price_precision': 2,
            'quantity_precision': 5,
            'additional_data': {
                'popular': True,
                'category': 'MAJOR',
                'description': 'Ethereum vs Tether',
                'is_stablecoin_pair': True
            }
        },
        {
            'symbol': 'ETH/BTC',
            'base_asset': 'ETH',
            'quote_asset': 'BTC',
            'min_order_size': 0.01,
            'max_order_size': 1000.0,
            'price_precision': 8,
            'quantity_precision': 5,
            'additional_data': {
                'popular': True,
                'category': 'CRYPTO_PAIR',
                'description': 'Ethereum vs Bitcoin',
                'is_stablecoin_pair': False
            }
        },
        {
            'symbol': 'SOL/USDT',
            'base_asset': 'SOL',
            'quote_asset': 'USDT',
            'min_order_size': 0.1,
            'max_order_size': 50000.0,
            'price_precision': 2,
            'quantity_precision': 2,
            'additional_data': {
                'popular': True,
                'category': 'ALTCOIN',
                'description': 'Solana vs Tether',
                'is_stablecoin_pair': True
            }
        },
        {
            'symbol': 'XRP/USDT',
            'base_asset': 'XRP',
            'quote_asset': 'USDT',
            'min_order_size': 1.0,
            'max_order_size': 100000.0,
            'price_precision': 5,
            'quantity_precision': 1,
            'additional_data': {
                'popular': True,
                'category': 'ALTCOIN',
                'description': 'Ripple vs Tether',
                'is_stablecoin_pair': True
            }
        }
    ]
    
    for data in trading_pairs:
        # Verificar si ya existe
        existing = session.query(DictTradingPair).filter_by(symbol=data['symbol']).first()
        if existing:
            logger.info(f"Par {data['symbol']} ya existe, actualizando...")
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            logger.info(f"Creando nuevo par de trading: {data['symbol']}")
            pair = DictTradingPair(
                symbol=data['symbol'],
                base_asset=data['base_asset'],
                quote_asset=data['quote_asset'],
                min_order_size=data['min_order_size'],
                max_order_size=data['max_order_size'],
                price_precision=data['price_precision'],
                quantity_precision=data['quantity_precision'],
                additional_data=data['additional_data'],
                is_active=True,
                created_at=datetime.now()
            )
            session.add(pair)
    
    session.commit()
    logger.info("Pares de trading inicializados correctamente.")

def init_risk_levels(session):
    """Inicializar niveles de riesgo."""
    logger.info("Inicializando niveles de riesgo...")
    
    risk_levels = [
        {
            'code': 'LOW',
            'name': 'Bajo',
            'description': 'Nivel de riesgo bajo, adecuado para estrategias conservadoras',
            'color': '#4CAF50',  # Verde
            'max_position_size_pct': 2.0,  # Máximo 2% del capital en una posición
            'max_drawdown_pct': 5.0,       # Máximo 5% de drawdown
            'risk_per_trade_pct': 0.5       # 0.5% de riesgo por operación
        },
        {
            'code': 'MEDIUM',
            'name': 'Medio',
            'description': 'Nivel de riesgo moderado, equilibrio entre seguridad y rendimiento',
            'color': '#FF9800',  # Naranja
            'max_position_size_pct': 5.0,  # Máximo 5% del capital en una posición
            'max_drawdown_pct': 15.0,      # Máximo 15% de drawdown
            'risk_per_trade_pct': 1.0       # 1% de riesgo por operación
        },
        {
            'code': 'HIGH',
            'name': 'Alto',
            'description': 'Nivel de riesgo alto, busca rendimientos elevados a costa de mayor volatilidad',
            'color': '#F44336',  # Rojo
            'max_position_size_pct': 10.0, # Máximo 10% del capital en una posición
            'max_drawdown_pct': 25.0,      # Máximo 25% de drawdown
            'risk_per_trade_pct': 2.0       # 2% de riesgo por operación
        },
        {
            'code': 'EXTREME',
            'name': 'Extremo',
            'description': 'Nivel de riesgo muy alto, solo para traders experimentados',
            'color': '#9C27B0',  # Púrpura
            'max_position_size_pct': 20.0, # Máximo 20% del capital en una posición
            'max_drawdown_pct': 40.0,      # Máximo 40% de drawdown
            'risk_per_trade_pct': 3.0       # 3% de riesgo por operación
        }
    ]
    
    for data in risk_levels:
        # Verificar si ya existe
        existing = session.query(DictRiskLevel).filter_by(code=data['code']).first()
        if existing:
            logger.info(f"Nivel de riesgo {data['code']} ya existe, actualizando...")
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            logger.info(f"Creando nuevo nivel de riesgo: {data['code']}")
            risk_level = DictRiskLevel(
                code=data['code'],
                name=data['name'],
                description=data['description'],
                color=data['color'],
                max_position_size_pct=data['max_position_size_pct'],
                max_drawdown_pct=data['max_drawdown_pct'],
                risk_per_trade_pct=data['risk_per_trade_pct'],
                is_active=True,
                created_at=datetime.now()
            )
            session.add(risk_level)
    
    session.commit()
    logger.info("Niveles de riesgo inicializados correctamente.")

def main():
    """Función principal para inicializar todas las tablas de diccionario."""
    logger.info(f"Conectando a la base de datos: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    
    # Crear tablas si no existen
    Base.metadata.create_all(engine)
    
    # Crear sesión
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Inicializar cada tipo de diccionario
        init_signal_types(session)
        init_trade_status(session)
        init_strategy_types(session)
        init_trading_pairs(session)
        init_risk_levels(session)
        
        logger.info("¡Inicialización de tablas de diccionario completada con éxito!")
    except Exception as e:
        logger.error(f"Error al inicializar tablas de diccionario: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()