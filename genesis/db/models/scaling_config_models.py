"""
Modelos de base de datos para la configuración de escalabilidad.

Este módulo define los modelos de base de datos necesarios para almacenar
configuraciones, puntos de saturación, historial de asignaciones y registros
de eficiencia para el sistema de escalabilidad adaptativa.
"""

import sqlalchemy as sa
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from typing import Dict, List, Any, Optional

Base = declarative_base()

class ScalingConfiguration(Base):
    """
    Configuración de escalabilidad para un conjunto de instrumentos o una estrategia.
    
    Esta tabla almacena los parámetros base para la escalabilidad adaptativa,
    incluyendo thresholds, límites y factores de ajuste.
    """
    __tablename__ = 'scaling_configurations'
    
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String(200), nullable=False)
    description = sa.Column(sa.Text, nullable=True)
    
    # Parámetros base
    capital_base = sa.Column(sa.Float, nullable=False, default=10000.0)
    efficiency_threshold = sa.Column(sa.Float, nullable=False, default=0.85)
    max_symbols_small = sa.Column(sa.Integer, nullable=False, default=5)
    max_symbols_large = sa.Column(sa.Integer, nullable=False, default=15)
    
    # Configuración avanzada
    volatility_adjustment = sa.Column(sa.Float, nullable=False, default=1.0)
    correlation_limit = sa.Column(sa.Float, nullable=False, default=0.7)
    capital_protection_level = sa.Column(sa.Float, nullable=False, default=0.95)
    
    # JSON con configuración adicional
    extended_config = sa.Column(JSONB, nullable=True)
    
    # Metadata
    created_at = sa.Column(sa.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, nullable=False, default=datetime.utcnow, 
                          onupdate=datetime.utcnow)
    active = sa.Column(sa.Boolean, nullable=False, default=True)
    
    # Relaciones
    saturation_points = relationship("SaturationPoint", back_populates="config",
                                    cascade="all, delete-orphan")
    allocation_history = relationship("AllocationHistory", back_populates="config",
                                     cascade="all, delete-orphan")
    efficiency_records = relationship("EfficiencyRecord", back_populates="config",
                                     cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ScalingConfiguration(id={self.id}, name='{self.name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capital_base": self.capital_base,
            "efficiency_threshold": self.efficiency_threshold,
            "max_symbols_small": self.max_symbols_small,
            "max_symbols_large": self.max_symbols_large,
            "volatility_adjustment": self.volatility_adjustment,
            "correlation_limit": self.correlation_limit,
            "capital_protection_level": self.capital_protection_level,
            "extended_config": self.extended_config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "active": self.active
        }


class SaturationPoint(Base):
    """
    Punto de saturación para un instrumento financiero.
    
    Representa el nivel de capital donde un instrumento comienza a mostrar
    deterioro de eficiencia debido a problemas de liquidez o impacto de mercado.
    """
    __tablename__ = 'saturation_points'
    
    id = sa.Column(sa.Integer, primary_key=True)
    config_id = sa.Column(sa.Integer, sa.ForeignKey('scaling_configurations.id'), 
                         nullable=False)
    symbol = sa.Column(sa.String(50), nullable=False)
    saturation_value = sa.Column(sa.Float, nullable=False)
    
    # Metadata sobre cómo se determinó este punto
    determination_method = sa.Column(sa.String(50), nullable=True)  # 'model', 'manual', 'historical'
    confidence = sa.Column(sa.Float, nullable=True)  # 0-1
    last_update = sa.Column(sa.DateTime, nullable=False, default=datetime.utcnow,
                           onupdate=datetime.utcnow)
    
    # Relación inversa
    config = relationship("ScalingConfiguration", back_populates="saturation_points")
    
    # Índice combinado para búsquedas rápidas
    __table_args__ = (
        sa.UniqueConstraint('config_id', 'symbol', name='uix_saturation_config_symbol'),
    )
    
    def __repr__(self):
        return f"<SaturationPoint(symbol='{self.symbol}', value={self.saturation_value:.2f})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "config_id": self.config_id,
            "symbol": self.symbol,
            "saturation_value": self.saturation_value,
            "determination_method": self.determination_method,
            "confidence": self.confidence,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }


class AllocationHistory(Base):
    """
    Historial de asignaciones de capital.
    
    Almacena un registro histórico de cómo se distribuyó el capital
    entre diferentes instrumentos a lo largo del tiempo.
    """
    __tablename__ = 'allocation_history'
    
    id = sa.Column(sa.Integer, primary_key=True)
    config_id = sa.Column(sa.Integer, sa.ForeignKey('scaling_configurations.id'), 
                         nullable=False)
    timestamp = sa.Column(sa.DateTime, nullable=False, default=datetime.utcnow)
    
    # Datos de asignación
    total_capital = sa.Column(sa.Float, nullable=False)
    scale_factor = sa.Column(sa.Float, nullable=False)
    instruments_count = sa.Column(sa.Integer, nullable=False)
    
    # Métricas de calidad
    capital_utilization = sa.Column(sa.Float, nullable=True)
    entropy = sa.Column(sa.Float, nullable=True)  # Diversificación
    efficiency_avg = sa.Column(sa.Float, nullable=True)
    
    # Datos detallados
    allocations = sa.Column(JSONB, nullable=False)  # {symbol: amount, ...}
    metrics = sa.Column(JSONB, nullable=True)  # Métricas adicionales
    
    # Relación inversa
    config = relationship("ScalingConfiguration", back_populates="allocation_history")
    
    def __repr__(self):
        return f"<AllocationHistory(id={self.id}, capital={self.total_capital:.2f}, instruments={self.instruments_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
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
    Registro de eficiencia observada para un instrumento.
    
    Almacena datos históricos de eficiencia observada para diferentes
    niveles de capital, permitiendo construir modelos predictivos.
    """
    __tablename__ = 'efficiency_records'
    
    id = sa.Column(sa.Integer, primary_key=True)
    config_id = sa.Column(sa.Integer, sa.ForeignKey('scaling_configurations.id'), 
                         nullable=False)
    symbol = sa.Column(sa.String(50), nullable=False)
    capital_level = sa.Column(sa.Float, nullable=False)
    efficiency = sa.Column(sa.Float, nullable=False)  # 0-1
    
    # Datos contextuales
    timestamp = sa.Column(sa.DateTime, nullable=False, default=datetime.utcnow)
    market_conditions = sa.Column(JSONB, nullable=True)  # Condiciones de mercado
    
    # Métricas de rendimiento
    roi = sa.Column(sa.Float, nullable=True)
    sharpe = sa.Column(sa.Float, nullable=True)
    max_drawdown = sa.Column(sa.Float, nullable=True)
    win_rate = sa.Column(sa.Float, nullable=True)
    
    # Relación inversa
    config = relationship("ScalingConfiguration", back_populates="efficiency_records")
    
    # Índices
    __table_args__ = (
        sa.Index('idx_efficiency_symbol_capital', 'symbol', 'capital_level'),
    )
    
    def __repr__(self):
        return f"<EfficiencyRecord(symbol='{self.symbol}', capital={self.capital_level:.2f}, efficiency={self.efficiency:.2f})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "config_id": self.config_id,
            "symbol": self.symbol,
            "capital_level": self.capital_level,
            "efficiency": self.efficiency,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "market_conditions": self.market_conditions,
            "roi": self.roi,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate
        }


class PredictiveModel(Base):
    """
    Modelo predictivo para estimación de eficiencia.
    
    Almacena modelos entrenados y sus parámetros para predecir
    cómo se comportarán diferentes instrumentos con distintos niveles de capital.
    """
    __tablename__ = 'predictive_models'
    
    id = sa.Column(sa.Integer, primary_key=True)
    config_id = sa.Column(sa.Integer, sa.ForeignKey('scaling_configurations.id'), 
                         nullable=False)
    symbol = sa.Column(sa.String(50), nullable=False)
    
    # Metadatos del modelo
    model_type = sa.Column(sa.String(50), nullable=False)  # linear, polynomial, exponential
    creation_date = sa.Column(sa.DateTime, nullable=False, default=datetime.utcnow)
    last_update = sa.Column(sa.DateTime, nullable=False, default=datetime.utcnow,
                           onupdate=datetime.utcnow)
    training_points = sa.Column(sa.Integer, nullable=False)
    
    # Parámetros del modelo
    parameters = sa.Column(JSONB, nullable=False)  # Coeficientes, hiperparámetros, etc.
    
    # Métricas de calidad
    r_squared = sa.Column(sa.Float, nullable=True)
    mean_error = sa.Column(sa.Float, nullable=True)
    max_error = sa.Column(sa.Float, nullable=True)
    
    # Rango válido
    valid_range_min = sa.Column(sa.Float, nullable=True)
    valid_range_max = sa.Column(sa.Float, nullable=True)
    
    # Índice combinado para búsquedas rápidas
    __table_args__ = (
        sa.UniqueConstraint('config_id', 'symbol', name='uix_model_config_symbol'),
    )
    
    def __repr__(self):
        return f"<PredictiveModel(symbol='{self.symbol}', type='{self.model_type}', r²={self.r_squared or 0:.3f})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "config_id": self.config_id,
            "symbol": self.symbol,
            "model_type": self.model_type,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "training_points": self.training_points,
            "parameters": self.parameters,
            "r_squared": self.r_squared,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "valid_range_min": self.valid_range_min,
            "valid_range_max": self.valid_range_max
        }