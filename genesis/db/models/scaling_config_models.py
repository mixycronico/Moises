"""
Modelos de configuración de escalabilidad para el Sistema Genesis.

Este módulo define los modelos de datos para la configuración del sistema de 
escalabilidad adaptativa, que permite mantener la eficiencia del sistema
cuando el capital crece significativamente.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime

from genesis.db.base import Base

class ScalingConfig(Base):
    """
    Configuración del sistema de escalabilidad adaptativa.
    
    Esta tabla almacena los parámetros de configuración para el gestor
    de escalabilidad de capital, que ajusta la distribución y parámetros
    según el nivel de fondos disponibles.
    """
    __tablename__ = "scaling_config"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    
    # Parámetros de escalabilidad
    capital_base = Column(Float, nullable=False, default=10000.0)
    efficiency_threshold = Column(Float, nullable=False, default=0.85)
    max_symbols_small = Column(Integer, nullable=False, default=5)
    max_symbols_large = Column(Integer, nullable=False, default=15)
    timeframes = Column(JSON, nullable=False, default=lambda: ["15m", "1h", "4h", "1d"])
    reallocation_interval_hours = Column(Integer, nullable=False, default=6)
    saturation_default = Column(Float, nullable=False, default=1000000.0)
    
    # Control de flujo
    active = Column(Boolean, nullable=False, default=True)
    redis_cache_enabled = Column(Boolean, nullable=False, default=True)
    monitoring_enabled = Column(Boolean, nullable=False, default=True)
    
    # Metadatos
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    
    # Configuración adicional en formato JSON
    advanced_config = Column(JSON, nullable=True)
    
    # Relaciones
    saturation_points = relationship("SaturationPoint", back_populates="config", cascade="all, delete-orphan")
    allocation_history = relationship("AllocationHistory", back_populates="config", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capital_base": self.capital_base,
            "efficiency_threshold": self.efficiency_threshold,
            "max_symbols_small": self.max_symbols_small,
            "max_symbols_large": self.max_symbols_large,
            "timeframes": self.timeframes,
            "reallocation_interval_hours": self.reallocation_interval_hours,
            "saturation_default": self.saturation_default,
            "active": self.active,
            "redis_cache_enabled": self.redis_cache_enabled,
            "monitoring_enabled": self.monitoring_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "advanced_config": self.advanced_config
        }


class SaturationPoint(Base):
    """
    Puntos de saturación estimados para instrumentos.
    
    Almacena los puntos de saturación calculados para cada instrumento,
    permitiendo persistir estos valores entre reinicios del sistema.
    """
    __tablename__ = "saturation_points"
    
    id = Column(Integer, primary_key=True)
    config_id = Column(Integer, ForeignKey("scaling_config.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    saturation_value = Column(Float, nullable=False)
    liquidity_score = Column(Float, nullable=True)
    volume_24h = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    
    # Metadatos
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    
    # Relaciones
    config = relationship("ScalingConfig", back_populates="saturation_points")
    
    __table_args__ = (
        # Índice compuesto para búsquedas rápidas
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )
    
    def to_dict(self):
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "config_id": self.config_id,
            "symbol": self.symbol,
            "saturation_value": self.saturation_value,
            "liquidity_score": self.liquidity_score,
            "volume_24h": self.volume_24h,
            "market_cap": self.market_cap,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class AllocationHistory(Base):
    """
    Historial de asignaciones de capital.
    
    Registra las asignaciones de capital realizadas a lo largo del tiempo,
    permitiendo análisis históricos y optimización continua.
    """
    __tablename__ = "allocation_history"
    
    id = Column(Integer, primary_key=True)
    config_id = Column(Integer, ForeignKey("scaling_config.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    
    # Datos de asignación
    total_capital = Column(Float, nullable=False)
    scale_factor = Column(Float, nullable=False)
    instruments_count = Column(Integer, nullable=False)
    
    # Métricas
    capital_utilization = Column(Float, nullable=False)
    entropy = Column(Float, nullable=True)
    efficiency_avg = Column(Float, nullable=True)
    
    # Datos detallados
    allocations = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=True)
    
    # Relaciones
    config = relationship("ScalingConfig", back_populates="allocation_history")
    
    __table_args__ = (
        # Índice para búsquedas por fecha
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )
    
    def to_dict(self):
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "config_id": self.config_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_capital": self.total_capital,
            "scale_factor": self.scale_factor,
            "instruments_count": self.instruments_count,
            "capital_utilization": self.capital_utilization,
            "entropy": self.entropy,
            "efficiency_avg": self.efficiency_avg,
            "allocations": self.allocations,
            "metrics": self.metrics
        }


class EfficiencyRecord(Base):
    """
    Registro de eficiencia por nivel de capital.
    
    Almacena las mediciones de eficiencia para diferentes niveles de
    capital por instrumento, permitiendo analizar el impacto del crecimiento
    de capital en la eficiencia operativa.
    """
    __tablename__ = "efficiency_records"
    
    id = Column(Integer, primary_key=True)
    config_id = Column(Integer, ForeignKey("scaling_config.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    
    # Datos de eficiencia
    capital_level = Column(Float, nullable=False)
    efficiency_score = Column(Float, nullable=False)
    
    # Datos adicionales
    roi = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    trades_count = Column(Integer, nullable=True)
    
    # Metadatos
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    
    # Relaciones
    config = relationship("ScalingConfig")
    
    __table_args__ = (
        # Índices para consultas rápidas
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )
    
    def to_dict(self):
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "config_id": self.config_id,
            "symbol": self.symbol,
            "capital_level": self.capital_level,
            "efficiency_score": self.efficiency_score,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "trades_count": self.trades_count,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }