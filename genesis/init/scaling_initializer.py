"""
Inicializador del sistema de escalabilidad adaptativa.

Este módulo se encarga de la inicialización del sistema de escalabilidad 
adaptativa, creando las tablas necesarias, cargando configuraciones 
predeterminadas y restaurando estados previos.
"""

import logging
import asyncio
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.db.models.scaling_config_models import (
    ScalingConfig, 
    SymbolScalingConfig, 
    SaturationPoint, 
    EfficiencyRecord,
    AllocationHistory, 
    ModelTrainingHistory,
    ModelType
)
from genesis.accounting.predictive_scaling import PredictiveScalingEngine
from genesis.accounting.balance_manager import CapitalScalingManager
from genesis.strategies.adaptive_scaling_strategy import AdaptiveScalingStrategy
from genesis.utils.helpers import generate_id

# Configurar logging
logger = logging.getLogger("genesis.init.scaling_initializer")

class ScalingInitializer:
    """
    Inicializador del sistema de escalabilidad adaptativa.
    
    Esta clase se encarga de:
    1. Crear las tablas necesarias en la base de datos
    2. Cargar configuraciones predeterminadas
    3. Inicializar los componentes de escalabilidad
    4. Restaurar estados previos si existen
    """
    
    def __init__(
        self,
        db: Optional[TranscendentalDatabase] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar el sistema de escalabilidad adaptativa.
        
        Args:
            db: Conexión a la base de datos transcendental
            config: Configuración adicional
        """
        self.db = db
        self.config = config or {}
        self.engine = None
        self.scaling_manager = None
        self.adaptive_strategy = None
        
        self.symbols = self.config.get('symbols', [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"
        ])
        
        logger.info(f"ScalingInitializer inicializado con {len(self.symbols)} símbolos")
    
    async def get_default_config_id(self) -> Optional[str]:
        """
        Obtener ID de la configuración predeterminada.
        
        Returns:
            ID de la configuración o None si no existe
        """
        if not self.db:
            return None
            
        try:
            result = await self.db.fetch_one(
                "SELECT id FROM scaling_config WHERE is_active = true ORDER BY created_at DESC LIMIT 1"
            )
            
            if result:
                return result['id']
                
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo configuración predeterminada: {str(e)}")
            return None
    
    async def create_default_config(self) -> Optional[str]:
        """
        Crear configuración predeterminada si no existe.
        
        Returns:
            ID de la configuración creada o None si falló
        """
        if not self.db:
            return None
            
        try:
            # Verificar si ya existe
            config_id = await self.get_default_config_id()
            if config_id:
                logger.info(f"Configuración predeterminada ya existe: {config_id}")
                return config_id
                
            # Crear nueva configuración
            config_id = generate_id()
            
            await self.db.execute(
                """
                INSERT INTO scaling_config 
                (id, name, description, default_model_type, min_efficiency_threshold, 
                rebalance_threshold, update_interval, optimization_method, 
                capital_reserve_percentage, min_position_size, max_position_percentage)
                VALUES 
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                config_id,
                "Configuración predeterminada",
                "Configuración generada automáticamente al inicializar el sistema",
                "polynomial",  # ModelType.POLYNOMIAL.value
                0.5,  # min_efficiency_threshold
                0.1,  # rebalance_threshold
                86400,  # update_interval (1 día)
                "marginal_utility",  # optimization_method
                0.05,  # capital_reserve_percentage
                0.0,  # min_position_size
                0.5   # max_position_percentage
            )
            
            logger.info(f"Configuración predeterminada creada: {config_id}")
            return config_id
            
        except Exception as e:
            logger.error(f"Error creando configuración predeterminada: {str(e)}")
            return None
    
    async def get_existing_symbols(self, config_id: str) -> List[str]:
        """
        Obtener símbolos ya configurados para una configuración.
        
        Args:
            config_id: ID de la configuración
            
        Returns:
            Lista de símbolos configurados
        """
        if not self.db or not config_id:
            return []
            
        try:
            count = await self.db.fetch_val(
                "SELECT COUNT(*) FROM symbol_scaling_config WHERE scaling_config_id = $1",
                config_id
            )
            
            if count == 0:
                return []
                
            results = await self.db.fetch(
                "SELECT symbol FROM symbol_scaling_config WHERE scaling_config_id = $1",
                config_id
            )
            
            return [row['symbol'] for row in results]
            
        except Exception as e:
            logger.error(f"Error obteniendo símbolos existentes: {str(e)}")
            return []
    
    async def initialize_components(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Inicializar componentes principales del sistema de escalabilidad.
        
        Returns:
            Tupla (éxito, componentes)
        """
        try:
            # 1. Inicializar el motor predictivo
            self.engine = PredictiveScalingEngine(
                config={
                    "default_model_type": self.config.get("model_type", "polynomial"),
                    "cache_ttl": self.config.get("cache_ttl", 300),
                    "auto_train": True,
                    "confidence_threshold": self.config.get("confidence_threshold", 0.6)
                }
            )
            
            # 2. Inicializar el scaling manager
            self.scaling_manager = CapitalScalingManager(
                capital_inicial=self.config.get("initial_capital", 10000.0),
                umbral_eficiencia=self.config.get("min_efficiency", 0.5)
            )
            # Asociar el motor predictivo y la BD al gestor de escalabilidad
            self.scaling_manager.engine = self.engine
            self.scaling_manager.db = self.db
            
            # 3. Cargar datos históricos si existen
            if self.db:
                symbols_str = ", ".join([f"'{s}'" for s in self.symbols])
                records = await self.db.fetch(
                    f"""
                    SELECT symbol, capital_level, efficiency, roi, sharpe, max_drawdown, win_rate
                    FROM efficiency_records
                    WHERE symbol IN ({symbols_str})
                    ORDER BY symbol, capital_level
                    """
                )
                
                # Cargar registros en el motor
                for record in records:
                    self.engine.add_efficiency_record(
                        symbol=record['symbol'],
                        capital=record['capital_level'],
                        efficiency=record['efficiency'],
                        metrics={
                            "roi": record.get('roi'),
                            "sharpe": record.get('sharpe'),
                            "max_drawdown": record.get('max_drawdown'),
                            "win_rate": record.get('win_rate')
                        }
                    )
            
            # 4. Inicializar la estrategia adaptativa
            self.adaptive_strategy = AdaptiveScalingStrategy(
                name="Estrategia de Escalabilidad Adaptativa",
                symbols=self.symbols,
                config=self.config,
                db=self.db
            )
            
            # 5. Vincular componentes
            self.adaptive_strategy.engine = self.engine
            self.adaptive_strategy.scaling_manager = self.scaling_manager
            
            # 6. Inicializar la estrategia
            await self.adaptive_strategy.initialize()
            
            return True, {
                "engine": self.engine,
                "scaling_manager": self.scaling_manager,
                "adaptive_strategy": self.adaptive_strategy
            }
            
        except Exception as e:
            logger.error(f"Error inicializando componentes: {str(e)}")
            return False, {}
    
    async def setup_database_tables(self) -> bool:
        """
        Configurar tablas de base de datos necesarias.
        
        Returns:
            True si la operación fue exitosa
        """
        if not self.db:
            logger.warning("No hay conexión a base de datos, saltando setup_database_tables")
            return False
            
        try:
            # Crear tabla scaling_config si no existe
            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS scaling_config (
                    id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL UNIQUE,
                    description VARCHAR(500),
                    default_model_type VARCHAR(50) NOT NULL,
                    min_efficiency_threshold FLOAT NOT NULL DEFAULT 0.5,
                    rebalance_threshold FLOAT NOT NULL DEFAULT 0.1,
                    update_interval INTEGER NOT NULL DEFAULT 86400,
                    optimization_method VARCHAR(50) DEFAULT 'marginal_utility',
                    capital_reserve_percentage FLOAT DEFAULT 0.05,
                    min_position_size FLOAT DEFAULT 0.0,
                    max_position_percentage FLOAT DEFAULT 0.5,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            # Crear tabla symbol_scaling_config si no existe
            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_scaling_config (
                    id VARCHAR(36) PRIMARY KEY,
                    scaling_config_id VARCHAR(36) NOT NULL REFERENCES scaling_config(id) ON DELETE CASCADE,
                    symbol VARCHAR(20) NOT NULL,
                    model_type VARCHAR(50),
                    model_parameters JSONB DEFAULT '{}',
                    min_efficiency_threshold FLOAT,
                    max_position_size FLOAT,
                    max_position_percentage FLOAT,
                    priority FLOAT DEFAULT 1.0,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(scaling_config_id, symbol)
                )
                """
            )
            
            # Crear tabla saturation_points si no existe
            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS saturation_points (
                    id VARCHAR(36) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL UNIQUE,
                    symbol_config_id VARCHAR(36) REFERENCES symbol_scaling_config(id) ON DELETE CASCADE,
                    saturation_value FLOAT NOT NULL,
                    efficiency_at_saturation FLOAT,
                    determination_method VARCHAR(50) DEFAULT 'model',
                    confidence FLOAT DEFAULT 0.5,
                    first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            # Crear tabla efficiency_records si no existe
            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS efficiency_records (
                    id VARCHAR(36) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    symbol_config_id VARCHAR(36) REFERENCES symbol_scaling_config(id) ON DELETE SET NULL,
                    capital_level FLOAT NOT NULL,
                    efficiency FLOAT NOT NULL,
                    roi FLOAT,
                    sharpe FLOAT,
                    max_drawdown FLOAT,
                    win_rate FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(50) DEFAULT 'system'
                )
                """
            )
            
            # Crear índice para efficiency_records
            await self.db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_efficiency_symbol_capital 
                ON efficiency_records(symbol, capital_level)
                """
            )
            
            # Crear tabla allocation_history si no existe
            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS allocation_history (
                    id VARCHAR(36) PRIMARY KEY,
                    scaling_config_id VARCHAR(36) REFERENCES scaling_config(id) ON DELETE CASCADE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_capital FLOAT NOT NULL,
                    allocations JSONB NOT NULL,
                    avg_efficiency FLOAT,
                    entropy FLOAT,
                    capital_utilization FLOAT,
                    rebalance_reason VARCHAR(50)
                )
                """
            )
            
            # Crear índice para allocation_history
            await self.db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_allocation_timestamp
                ON allocation_history(timestamp)
                """
            )
            
            # Crear tabla model_training_history si no existe
            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS model_training_history (
                    id VARCHAR(36) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_type VARCHAR(50) NOT NULL,
                    parameters JSONB NOT NULL,
                    r_squared FLOAT,
                    mean_error FLOAT,
                    max_error FLOAT,
                    samples_count INTEGER,
                    detected_saturation FLOAT
                )
                """
            )
            
            # Crear índice para model_training_history
            await self.db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model_training_symbol
                ON model_training_history(symbol)
                """
            )
            
            logger.info("Tablas de base de datos configuradas correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error configurando tablas de base de datos: {str(e)}")
            return False
    
    async def setup_symbol_configs(self, config_id: str) -> bool:
        """
        Configurar símbolos para una configuración específica.
        
        Args:
            config_id: ID de la configuración
            
        Returns:
            True si la operación fue exitosa
        """
        if not self.db or not config_id:
            return False
            
        try:
            # Obtener símbolos ya configurados
            existing_symbols = await self.get_existing_symbols(config_id)
            
            # Determinar símbolos a añadir
            new_symbols = [s for s in self.symbols if s not in existing_symbols]
            
            if not new_symbols:
                logger.info("No hay nuevos símbolos para configurar")
                return True
                
            # Añadir nuevos símbolos
            for symbol in new_symbols:
                symbol_id = generate_id()
                
                await self.db.execute(
                    """
                    INSERT INTO symbol_scaling_config
                    (id, scaling_config_id, symbol, model_type, priority)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    symbol_id,
                    config_id,
                    symbol,
                    "polynomial",  # Por defecto
                    1.0  # Prioridad estándar
                )
                
            logger.info(f"Configurados {len(new_symbols)} nuevos símbolos")
            return True
            
        except Exception as e:
            logger.error(f"Error configurando símbolos: {str(e)}")
            return False
    
    async def initialize(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Inicializar el sistema de escalabilidad adaptativa completo.
        
        Returns:
            Tupla (éxito, componentes)
        """
        try:
            logger.info("Iniciando inicialización del sistema de escalabilidad adaptativa")
            
            # 1. Configurar tablas de base de datos
            if self.db:
                db_setup = await self.setup_database_tables()
                if not db_setup:
                    logger.warning("Configuración de base de datos fallida")
                
                # 2. Crear configuración predeterminada
                config_id = await self.create_default_config()
                if config_id:
                    # 3. Configurar símbolos
                    await self.setup_symbol_configs(config_id)
            
            # 4. Inicializar componentes
            success, components = await self.initialize_components()
            
            if success:
                logger.info("Sistema de escalabilidad adaptativa inicializado correctamente")
            else:
                logger.error("Fallo en la inicialización de componentes")
                
            return success, components
            
        except Exception as e:
            logger.error(f"Error en la inicialización: {str(e)}")
            return False, {}