"""
Inicialización del sistema de escalabilidad adaptativa.

Este módulo inicializa el motor de escalabilidad adaptativa y carga
la configuración necesaria desde la base de datos.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from genesis.db.transcendental_database import TranscendentalDatabase
from genesis.db.models.scaling_config_models import ScalingConfiguration
from genesis.accounting.balance_manager import CapitalScalingManager
from genesis.accounting.predictive_scaling import PredictiveScalingEngine
from genesis.utils.config import get_config

class ScalingInitializer:
    """
    Inicializador del sistema de escalabilidad adaptativa.
    
    Este componente se encarga de inicializar el motor de escalabilidad,
    cargar configuraciones desde la base de datos y preparar el sistema
    para su uso.
    """
    
    def __init__(self, db: Optional[TranscendentalDatabase] = None):
        """
        Inicializar el inicializador de escalabilidad.
        
        Args:
            db: Instancia de la base de datos transcendental
        """
        self.logger = logging.getLogger('genesis.init.scaling_initializer')
        self.db = db
        self.config = get_config('scaling')
        
    async def load_default_config(self) -> ScalingConfiguration:
        """
        Cargar configuración por defecto desde la base de datos.
        
        Si no existe una configuración por defecto, la crea.
        
        Returns:
            Configuración de escalabilidad
        """
        if not self.db:
            # Crear configuración en memoria
            return ScalingConfiguration(
                id=1,
                name="Default",
                description="Configuración por defecto",
                capital_base=10000.0,
                efficiency_threshold=0.85,
                max_symbols_small=5,
                max_symbols_large=15,
                volatility_adjustment=1.0,
                correlation_limit=0.7,
                capital_protection_level=0.95
            )
            
        # Buscar configuración por defecto
        config_row = await self.db.fetch_one(
            "SELECT * FROM scaling_configurations WHERE active = true ORDER BY id LIMIT 1"
        )
        
        if config_row:
            self.logger.info(f"Configuración de escalabilidad cargada: {config_row['name']}")
            return ScalingConfiguration(**config_row)
        else:
            # Crear configuración por defecto
            self.logger.info("Creando configuración de escalabilidad por defecto")
            default_config = ScalingConfiguration(
                name="Default",
                description="Configuración por defecto",
                capital_base=10000.0,
                efficiency_threshold=0.85,
                max_symbols_small=5,
                max_symbols_large=15,
                volatility_adjustment=1.0,
                correlation_limit=0.7,
                capital_protection_level=0.95
            )
            
            # Guardar en la base de datos
            await self.db.execute(
                "INSERT INTO scaling_configurations (name, description, capital_base, "
                "efficiency_threshold, max_symbols_small, max_symbols_large, "
                "volatility_adjustment, correlation_limit, capital_protection_level, "
                "created_at, updated_at, active) "
                "VALUES (:name, :description, :capital_base, :efficiency_threshold, "
                ":max_symbols_small, :max_symbols_large, :volatility_adjustment, "
                ":correlation_limit, :capital_protection_level, NOW(), NOW(), :active)",
                default_config.to_dict()
            )
            
            # Obtener el ID generado
            config_id = await self.db.fetch_val(
                "SELECT id FROM scaling_configurations WHERE name = :name ORDER BY id DESC LIMIT 1",
                {"name": default_config.name}
            )
            
            default_config.id = config_id
            return default_config
            
    async def initialize_scaling_manager(self) -> CapitalScalingManager:
        """
        Inicializar el gestor de escalabilidad.
        
        Returns:
            Instancia de CapitalScalingManager inicializada
        """
        # Cargar configuración
        config = await self.load_default_config()
        
        # Crear motor predictivo
        predictive_engine = PredictiveScalingEngine(
            config={
                "default_model_type": "polynomial",
                "cache_ttl": 300,
                "auto_train": True,
                "confidence_threshold": 0.7
            }
        )
        
        # Convertir configuración a diccionario
        config_dict = {
            "id": config.id,
            "name": config.name,
            "capital_base": config.capital_base,
            "efficiency_threshold": config.efficiency_threshold,
            "max_symbols_small": config.max_symbols_small,
            "max_symbols_large": config.max_symbols_large,
            "volatility_adjustment": config.volatility_adjustment,
            "correlation_limit": config.correlation_limit,
            "capital_protection_level": config.capital_protection_level,
            "db_persistence_enabled": True if self.db else False,
            "monitoring_enabled": True,
            "redis_cache_enabled": True,
            "max_capital": config.capital_base * 100  # 100x el capital base como límite
        }
        
        # Crear gestor de escalabilidad
        scaling_manager = CapitalScalingManager(
            config=config_dict,
            predictive_engine=predictive_engine
        )
        
        # Asignar base de datos
        scaling_manager.db = self.db
        
        # Cargar datos históricos si están disponibles
        if self.db:
            await scaling_manager.load_saturation_points()
            
            # Cargar registros de eficiencia históricos
            rows = await self.db.fetch(
                "SELECT symbol, capital_level, efficiency FROM efficiency_records "
                "WHERE config_id = :config_id",
                {"config_id": config.id}
            )
            
            if rows:
                for row in rows:
                    predictive_engine.add_efficiency_record(
                        symbol=row['symbol'],
                        capital=row['capital_level'],
                        efficiency=row['efficiency']
                    )
                
                self.logger.info(f"Cargados {len(rows)} registros de eficiencia históricos")
                
        self.logger.info(f"Gestor de escalabilidad inicializado con capital base {config.capital_base}")
        return scaling_manager
    
    async def create_database_tables(self) -> None:
        """
        Crear tablas en la base de datos.
        
        Este método crea las tablas necesarias si no existen.
        """
        if not self.db:
            self.logger.warning("No hay base de datos disponible para crear tablas")
            return
            
        try:
            # Crear tablas de configuración de escalabilidad
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS scaling_configurations (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    capital_base FLOAT NOT NULL DEFAULT 10000.0,
                    efficiency_threshold FLOAT NOT NULL DEFAULT 0.85,
                    max_symbols_small INTEGER NOT NULL DEFAULT 5,
                    max_symbols_large INTEGER NOT NULL DEFAULT 15,
                    volatility_adjustment FLOAT NOT NULL DEFAULT 1.0,
                    correlation_limit FLOAT NOT NULL DEFAULT 0.7,
                    capital_protection_level FLOAT NOT NULL DEFAULT 0.95,
                    extended_config JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    active BOOLEAN NOT NULL DEFAULT TRUE
                )
            """)
            
            # Tabla de puntos de saturación
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS saturation_points (
                    id SERIAL PRIMARY KEY,
                    config_id INTEGER NOT NULL REFERENCES scaling_configurations(id),
                    symbol VARCHAR(50) NOT NULL,
                    saturation_value FLOAT NOT NULL,
                    determination_method VARCHAR(50),
                    confidence FLOAT,
                    last_update TIMESTAMP NOT NULL DEFAULT NOW(),
                    CONSTRAINT uix_saturation_config_symbol UNIQUE (config_id, symbol)
                )
            """)
            
            # Tabla de historial de asignaciones
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS allocation_history (
                    id SERIAL PRIMARY KEY,
                    config_id INTEGER NOT NULL REFERENCES scaling_configurations(id),
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    total_capital FLOAT NOT NULL,
                    scale_factor FLOAT NOT NULL,
                    instruments_count INTEGER NOT NULL,
                    capital_utilization FLOAT,
                    entropy FLOAT,
                    efficiency_avg FLOAT,
                    allocations JSONB NOT NULL,
                    metrics JSONB
                )
            """)
            
            # Tabla de registros de eficiencia
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS efficiency_records (
                    id SERIAL PRIMARY KEY,
                    config_id INTEGER NOT NULL REFERENCES scaling_configurations(id),
                    symbol VARCHAR(50) NOT NULL,
                    capital_level FLOAT NOT NULL,
                    efficiency FLOAT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    market_conditions JSONB,
                    roi FLOAT,
                    sharpe FLOAT,
                    max_drawdown FLOAT,
                    win_rate FLOAT
                )
            """)
            
            # Tabla de modelos predictivos
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS predictive_models (
                    id SERIAL PRIMARY KEY,
                    config_id INTEGER NOT NULL REFERENCES scaling_configurations(id),
                    symbol VARCHAR(50) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    creation_date TIMESTAMP NOT NULL DEFAULT NOW(),
                    last_update TIMESTAMP NOT NULL DEFAULT NOW(),
                    training_points INTEGER NOT NULL,
                    parameters JSONB NOT NULL,
                    r_squared FLOAT,
                    mean_error FLOAT,
                    max_error FLOAT,
                    valid_range_min FLOAT,
                    valid_range_max FLOAT,
                    CONSTRAINT uix_model_config_symbol UNIQUE (config_id, symbol)
                )
            """)
            
            # Índices adicionales para mejor rendimiento
            await self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_efficiency_symbol_capital 
                ON efficiency_records (symbol, capital_level)
            """)
            
            self.logger.info("Tablas de escalabilidad creadas correctamente")
            
        except Exception as e:
            self.logger.error(f"Error creando tablas de escalabilidad: {str(e)}")
            raise