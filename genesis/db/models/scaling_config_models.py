"""
Modelos de base de datos para el sistema de escalabilidad adaptativa.

Este módulo define los modelos y esquemas de datos utilizados para 
configurar y almacenar información de la escalabilidad adaptativa, 
incluyendo puntos de saturación, registros de eficiencia, y 
configuraciones de predicción.
"""

import sqlalchemy as sa
import sqlalchemy.orm as orm
import sqlalchemy.ext.hybrid as hybrid
import sqlalchemy.dialects.postgresql as pg
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum as PyEnum

from genesis.db.base import Base
from genesis.utils.helpers import generate_id

class ModelType(PyEnum):
    """Tipos de modelos predictivos disponibles."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    CUSTOM = "custom"


class ScalingConfig(Base):
    """
    Configuración general del sistema de escalabilidad adaptativa.
    
    Almacena parámetros globales como tipo de modelo por defecto,
    umbrales de rebalanceo, y configuraciones de optimización.
    """
    __tablename__ = "scaling_config"
    
    id = sa.Column(sa.String(36), primary_key=True, default=generate_id)
    name = sa.Column(sa.String(100), nullable=False, unique=True)
    description = sa.Column(sa.String(500))
    
    # Parámetros globales
    default_model_type = sa.Column(sa.Enum(ModelType), nullable=False, 
                                 default=ModelType.POLYNOMIAL)
    min_efficiency_threshold = sa.Column(sa.Float, nullable=False, default=0.5)
    rebalance_threshold = sa.Column(sa.Float, nullable=False, default=0.1)
    update_interval = sa.Column(sa.Integer, nullable=False, default=86400)  # en segundos
    
    # Parámetros de optimización
    optimization_method = sa.Column(sa.String(50), default="marginal_utility")
    capital_reserve_percentage = sa.Column(sa.Float, default=0.05)  # 5% por defecto
    min_position_size = sa.Column(sa.Float, default=0.0)
    max_position_percentage = sa.Column(sa.Float, default=0.5)  # 50% por defecto
    
    # Metadatos
    is_active = sa.Column(sa.Boolean, default=True)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, 
                          onupdate=datetime.utcnow)
    
    # Relaciones
    symbol_configs = orm.relationship("SymbolScalingConfig", 
                                    back_populates="scaling_config",
                                    cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para APIs."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default_model_type": self.default_model_type.value,
            "min_efficiency_threshold": self.min_efficiency_threshold,
            "rebalance_threshold": self.rebalance_threshold,
            "update_interval": self.update_interval,
            "optimization_method": self.optimization_method,
            "capital_reserve_percentage": self.capital_reserve_percentage,
            "min_position_size": self.min_position_size,
            "max_position_percentage": self.max_position_percentage,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SymbolScalingConfig(Base):
    """
    Configuración específica de escalabilidad para un símbolo/instrumento.
    
    Permite sobrescribir configuraciones globales para instrumentos específicos,
    como tipos de modelo, parámetros, y restricciones.
    """
    __tablename__ = "symbol_scaling_config"
    
    id = sa.Column(sa.String(36), primary_key=True, default=generate_id)
    scaling_config_id = sa.Column(sa.String(36), 
                               sa.ForeignKey("scaling_config.id", ondelete="CASCADE"),
                               nullable=False)
    symbol = sa.Column(sa.String(20), nullable=False)
    
    # Configuración específica del símbolo
    model_type = sa.Column(sa.Enum(ModelType))
    model_parameters = sa.Column(pg.JSONB, default={})
    min_efficiency_threshold = sa.Column(sa.Float)
    max_position_size = sa.Column(sa.Float)  # Valor absoluto
    max_position_percentage = sa.Column(sa.Float)  # Porcentaje del capital total
    priority = sa.Column(sa.Float, default=1.0)  # Prioridad en optimizaciones
    
    # Metadatos
    is_active = sa.Column(sa.Boolean, default=True)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, 
                          onupdate=datetime.utcnow)
    
    # Relaciones
    scaling_config = orm.relationship("ScalingConfig", back_populates="symbol_configs")
    saturation_point = orm.relationship("SaturationPoint", 
                                      uselist=False, 
                                      back_populates="symbol_config",
                                      cascade="all, delete-orphan")
    efficiency_records = orm.relationship("EfficiencyRecord", 
                                        back_populates="symbol_config",
                                        cascade="all, delete-orphan")
    
    # Índices
    __table_args__ = (
        sa.UniqueConstraint('scaling_config_id', 'symbol', 
                         name='uix_symbol_config'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para APIs."""
        return {
            "id": self.id,
            "scaling_config_id": self.scaling_config_id,
            "symbol": self.symbol,
            "model_type": self.model_type.value if self.model_type else None,
            "model_parameters": self.model_parameters,
            "min_efficiency_threshold": self.min_efficiency_threshold,
            "max_position_size": self.max_position_size,
            "max_position_percentage": self.max_position_percentage,
            "priority": self.priority,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SaturationPoint(Base):
    """
    Punto de saturación detectado para un instrumento.
    
    El punto de saturación es el nivel de capital donde el instrumento
    comienza a perder eficiencia significativamente.
    """
    __tablename__ = "saturation_points"
    
    id = sa.Column(sa.String(36), primary_key=True, default=generate_id)
    symbol = sa.Column(sa.String(20), nullable=False, unique=True)
    symbol_config_id = sa.Column(sa.String(36), 
                              sa.ForeignKey("symbol_scaling_config.id", ondelete="CASCADE"))
    
    # Datos de saturación
    saturation_value = sa.Column(sa.Float, nullable=False)  # Nivel de capital
    efficiency_at_saturation = sa.Column(sa.Float)  # Eficiencia en ese punto
    determination_method = sa.Column(sa.String(50), default="model")  # Cómo se determinó
    confidence = sa.Column(sa.Float, default=0.5)  # Confianza en la detección (0-1)
    
    # Metadatos
    first_detected = sa.Column(sa.DateTime, default=datetime.utcnow)
    last_update = sa.Column(sa.DateTime, default=datetime.utcnow, 
                           onupdate=datetime.utcnow)
    
    # Relaciones
    symbol_config = orm.relationship("SymbolScalingConfig", back_populates="saturation_point")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para APIs."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "saturation_value": self.saturation_value,
            "efficiency_at_saturation": self.efficiency_at_saturation,
            "determination_method": self.determination_method,
            "confidence": self.confidence,
            "first_detected": self.first_detected.isoformat() if self.first_detected else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


class EfficiencyRecord(Base):
    """
    Registro de eficiencia observada para un instrumento a un nivel de capital.
    
    Estos registros alimentan los modelos predictivos para mejorar sus predicciones.
    """
    __tablename__ = "efficiency_records"
    
    id = sa.Column(sa.String(36), primary_key=True, default=generate_id)
    symbol = sa.Column(sa.String(20), nullable=False, index=True)
    symbol_config_id = sa.Column(sa.String(36), 
                              sa.ForeignKey("symbol_scaling_config.id", ondelete="SET NULL"),
                              nullable=True)
    
    # Datos de eficiencia
    capital_level = sa.Column(sa.Float, nullable=False)  # Nivel de capital
    efficiency = sa.Column(sa.Float, nullable=False)  # Eficiencia observada (0-1)
    
    # Métricas adicionales
    roi = sa.Column(sa.Float)  # Retorno de inversión
    sharpe = sa.Column(sa.Float)  # Ratio de Sharpe
    max_drawdown = sa.Column(sa.Float)  # Drawdown máximo
    win_rate = sa.Column(sa.Float)  # Tasa de operaciones ganadoras
    
    # Metadatos
    timestamp = sa.Column(sa.DateTime, default=datetime.utcnow)
    source = sa.Column(sa.String(50), default="system")  # Origen del registro
    
    # Relaciones
    symbol_config = orm.relationship("SymbolScalingConfig", back_populates="efficiency_records")
    
    # Índices
    __table_args__ = (
        sa.Index('idx_efficiency_symbol_capital', 'symbol', 'capital_level'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para APIs."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "capital_level": self.capital_level,
            "efficiency": self.efficiency,
            "roi": self.roi,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
        }


class AllocationHistory(Base):
    """
    Historial de asignaciones de capital realizadas por el sistema.
    
    Permite analizar cómo han evolucionado las asignaciones a lo largo
    del tiempo y evaluar la efectividad de las optimizaciones.
    """
    __tablename__ = "allocation_history"
    
    id = sa.Column(sa.String(36), primary_key=True, default=generate_id)
    scaling_config_id = sa.Column(sa.String(36), 
                               sa.ForeignKey("scaling_config.id", ondelete="CASCADE"),
                               nullable=True)
    
    # Datos de asignación
    timestamp = sa.Column(sa.DateTime, default=datetime.utcnow, index=True)
    total_capital = sa.Column(sa.Float, nullable=False)
    allocations = sa.Column(pg.JSONB, nullable=False)  # {"symbol": amount, ...}
    
    # Métricas
    avg_efficiency = sa.Column(sa.Float)
    entropy = sa.Column(sa.Float)  # Diversificación
    capital_utilization = sa.Column(sa.Float)  # % de capital utilizado
    rebalance_reason = sa.Column(sa.String(50))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para APIs."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_capital": self.total_capital,
            "allocations": self.allocations,
            "avg_efficiency": self.avg_efficiency,
            "entropy": self.entropy,
            "capital_utilization": self.capital_utilization,
            "rebalance_reason": self.rebalance_reason,
        }


class ModelTrainingHistory(Base):
    """
    Historial de entrenamiento de modelos predictivos.
    
    Registra cómo han evolucionado los modelos a lo largo del tiempo,
    sus parámetros y métricas de calidad.
    """
    __tablename__ = "model_training_history"
    
    id = sa.Column(sa.String(36), primary_key=True, default=generate_id)
    symbol = sa.Column(sa.String(20), nullable=False, index=True)
    
    # Datos de entrenamiento
    timestamp = sa.Column(sa.DateTime, default=datetime.utcnow)
    model_type = sa.Column(sa.Enum(ModelType), nullable=False)
    parameters = sa.Column(pg.JSONB, nullable=False)  # Parámetros del modelo
    
    # Métricas de calidad
    r_squared = sa.Column(sa.Float)  # Coeficiente de determinación
    mean_error = sa.Column(sa.Float)  # Error medio
    max_error = sa.Column(sa.Float)  # Error máximo
    samples_count = sa.Column(sa.Integer)  # Número de muestras usadas
    
    # Saturation point detectado
    detected_saturation = sa.Column(sa.Float)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para APIs."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_type": self.model_type.value,
            "parameters": self.parameters,
            "r_squared": self.r_squared,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "samples_count": self.samples_count,
            "detected_saturation": self.detected_saturation,
        }