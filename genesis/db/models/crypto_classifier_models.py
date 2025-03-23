"""
Modelos para el clasificador transcendental de criptomonedas.

Este módulo define los modelos SQLAlchemy para el sistema de clasificación
adaptativa de criptomonedas, permitiendo almacenar y consultar eficientemente
datos históricos de clasificaciones, métricas y rendimiento.
"""
import datetime
from typing import Optional, Dict, Any, List

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from genesis.db.base import Base


class Cryptocurrency(Base):
    """Modelo para representar criptomonedas."""
    
    __tablename__ = "cryptocurrencies"
    
    id = sa.Column(sa.Integer, primary_key=True)
    symbol = sa.Column(sa.String(20), nullable=False, unique=True, index=True)
    name = sa.Column(sa.String(100), nullable=False)
    
    # Datos de mercado
    market_cap = sa.Column(sa.BigInteger, nullable=True)
    volume_24h = sa.Column(sa.BigInteger, nullable=True)
    circulating_supply = sa.Column(sa.BigInteger, nullable=True)
    max_supply = sa.Column(sa.BigInteger, nullable=True)
    
    # Datos técnicos
    current_price = sa.Column(sa.Float, nullable=True)
    price_change_24h = sa.Column(sa.Float, nullable=True)
    price_change_percentage_24h = sa.Column(sa.Float, nullable=True)
    
    # Metadatos
    updated_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    created_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)
    is_active = sa.Column(sa.Boolean, default=True)
    
    # Relaciones
    classifications = relationship("CryptoClassification", back_populates="cryptocurrency")
    metrics = relationship("CryptoMetrics", back_populates="cryptocurrency")
    
    def __repr__(self):
        return f"<Cryptocurrency {self.symbol} ({self.name})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "name": self.name,
            "market_cap": self.market_cap,
            "volume_24h": self.volume_24h,
            "circulating_supply": self.circulating_supply,
            "max_supply": self.max_supply,
            "current_price": self.current_price,
            "price_change_24h": self.price_change_24h,
            "price_change_percentage_24h": self.price_change_percentage_24h,
            "is_active": self.is_active,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class CryptoClassification(Base):
    """Modelo para clasificaciones de criptomonedas."""
    
    __tablename__ = "crypto_classifications"
    
    id = sa.Column(sa.Integer, primary_key=True)
    cryptocurrency_id = sa.Column(sa.Integer, sa.ForeignKey("cryptocurrencies.id"), nullable=False)
    
    # Factores de clasificación
    alpha_score = sa.Column(sa.Float, nullable=False)  # Rentabilidad ajustada por riesgo
    liquidity_score = sa.Column(sa.Float, nullable=False)  # Liquidez del mercado
    volatility_score = sa.Column(sa.Float, nullable=False)  # Volatilidad histórica normalizada
    momentum_score = sa.Column(sa.Float, nullable=False)  # Impulso de precio reciente
    trend_score = sa.Column(sa.Float, nullable=False)  # Fuerza de la tendencia actual
    correlation_score = sa.Column(sa.Float, nullable=False)  # Correlación con el mercado
    exchange_quality_score = sa.Column(sa.Float, nullable=False)  # Calidad de los exchanges soportados
    
    # Calificación final
    final_score = sa.Column(sa.Float, nullable=False)  # Puntuación final combinada
    hot_rating = sa.Column(sa.Boolean, default=False)  # Si es una criptomoneda "hot"
    
    # Configuración utilizada
    capital_base = sa.Column(sa.Float, nullable=False)  # Base de capital usada para clasificación
    
    # Metadatos
    classification_date = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)
    confidence = sa.Column(sa.Float, default=0.0)  # Confianza en la clasificación (0-1)
    
    # Relaciones
    cryptocurrency = relationship("Cryptocurrency", back_populates="classifications")
    
    def __repr__(self):
        return f"<CryptoClassification {self.cryptocurrency.symbol if self.cryptocurrency else 'Unknown'} score={self.final_score:.2f}>"
    
    @hybrid_property
    def is_hot(self) -> bool:
        """Determinar si es una criptomoneda 'hot' basada en umbral dinámico."""
        return self.hot_rating
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "cryptocurrency_id": self.cryptocurrency_id,
            "alpha_score": self.alpha_score,
            "liquidity_score": self.liquidity_score,
            "volatility_score": self.volatility_score,
            "momentum_score": self.momentum_score,
            "trend_score": self.trend_score,
            "correlation_score": self.correlation_score,
            "exchange_quality_score": self.exchange_quality_score,
            "final_score": self.final_score,
            "hot_rating": self.hot_rating,
            "capital_base": self.capital_base,
            "classification_date": self.classification_date.isoformat() if self.classification_date else None,
            "confidence": self.confidence,
            "is_hot": self.is_hot
        }


class CryptoMetrics(Base):
    """Modelo para métricas de criptomonedas."""
    
    __tablename__ = "crypto_metrics"
    
    id = sa.Column(sa.Integer, primary_key=True)
    cryptocurrency_id = sa.Column(sa.Integer, sa.ForeignKey("cryptocurrencies.id"), nullable=False)
    
    # Métricas avanzadas
    orderbook_depth_usd = sa.Column(sa.Float, nullable=True)  # Profundidad del libro de órdenes (USD)
    slippage_1000usd = sa.Column(sa.Float, nullable=True)  # Deslizamiento para orden de $1000
    slippage_10000usd = sa.Column(sa.Float, nullable=True)  # Deslizamiento para orden de $10000
    slippage_100000usd = sa.Column(sa.Float, nullable=True)  # Deslizamiento para orden de $100000
    
    # Métricas on-chain (si están disponibles)
    active_addresses = sa.Column(sa.Integer, nullable=True)
    transaction_count_24h = sa.Column(sa.Integer, nullable=True)
    network_hash_rate = sa.Column(sa.BigInteger, nullable=True)
    
    # Métricas sociales
    social_volume = sa.Column(sa.Float, nullable=True)  # Volumen de menciones sociales
    sentiment_score = sa.Column(sa.Float, nullable=True)  # Sentimiento general (-1 a 1)
    developer_activity = sa.Column(sa.Float, nullable=True)  # Actividad de desarrollo
    
    # Métricas de desempeño
    sharpe_ratio = sa.Column(sa.Float, nullable=True)  # Ratio de Sharpe (30 días)
    sortino_ratio = sa.Column(sa.Float, nullable=True)  # Ratio de Sortino (30 días)
    volatility_30d = sa.Column(sa.Float, nullable=True)  # Volatilidad (30 días)
    drawdown_max = sa.Column(sa.Float, nullable=True)  # Drawdown máximo histórico
    
    # Metadatos
    updated_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    source = sa.Column(sa.String(50), nullable=True)  # Fuente de los datos
    
    # Relaciones
    cryptocurrency = relationship("Cryptocurrency", back_populates="metrics")
    
    def __repr__(self):
        return f"<CryptoMetrics {self.cryptocurrency.symbol if self.cryptocurrency else 'Unknown'}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "cryptocurrency_id": self.cryptocurrency_id,
            "orderbook_depth_usd": self.orderbook_depth_usd,
            "slippage_1000usd": self.slippage_1000usd,
            "slippage_10000usd": self.slippage_10000usd,
            "slippage_100000usd": self.slippage_100000usd,
            "active_addresses": self.active_addresses,
            "transaction_count_24h": self.transaction_count_24h,
            "network_hash_rate": self.network_hash_rate,
            "social_volume": self.social_volume,
            "sentiment_score": self.sentiment_score,
            "developer_activity": self.developer_activity,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "volatility_30d": self.volatility_30d,
            "drawdown_max": self.drawdown_max,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source": self.source
        }


class ClassificationHistory(Base):
    """Modelo para seguimiento histórico de cambios en clasificaciones."""
    
    __tablename__ = "classification_history"
    
    id = sa.Column(sa.Integer, primary_key=True)
    classification_id = sa.Column(sa.Integer, sa.ForeignKey("crypto_classifications.id"), nullable=False)
    change_date = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)
    
    # Cambios registrados
    previous_final_score = sa.Column(sa.Float, nullable=True)
    new_final_score = sa.Column(sa.Float, nullable=False)
    
    previous_hot_rating = sa.Column(sa.Boolean, nullable=True)
    new_hot_rating = sa.Column(sa.Boolean, nullable=False)
    
    # Contexto del cambio
    capital_base = sa.Column(sa.Float, nullable=False)  # Capital base en el momento del cambio
    market_condition = sa.Column(sa.String(20), nullable=True)  # Ej. "bull", "bear", "neutral"
    
    # Análisis del cambio
    change_magnitude = sa.Column(sa.Float, nullable=False)  # Magnitud del cambio (porcentaje)
    change_reason = sa.Column(sa.String(200), nullable=True)  # Razón para el cambio
    
    def __repr__(self):
        return f"<ClassificationHistory id={self.id} change={self.change_magnitude:.2f}%>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "classification_id": self.classification_id,
            "change_date": self.change_date.isoformat() if self.change_date else None,
            "previous_final_score": self.previous_final_score,
            "new_final_score": self.new_final_score,
            "previous_hot_rating": self.previous_hot_rating,
            "new_hot_rating": self.new_hot_rating,
            "capital_base": self.capital_base,
            "market_condition": self.market_condition,
            "change_magnitude": self.change_magnitude,
            "change_reason": self.change_reason
        }


class CapitalScaleEffect(Base):
    """Modelo para el análisis del efecto del tamaño del capital en las clasificaciones."""
    
    __tablename__ = "capital_scale_effects"
    
    id = sa.Column(sa.Integer, primary_key=True)
    cryptocurrency_id = sa.Column(sa.Integer, sa.ForeignKey("cryptocurrencies.id"), nullable=False)
    analysis_date = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)
    
    # Puntuaciones para diferentes niveles de capital
    score_10k = sa.Column(sa.Float, nullable=False)  # Score con $10,000
    score_100k = sa.Column(sa.Float, nullable=False)  # Score con $100,000
    score_1m = sa.Column(sa.Float, nullable=False)  # Score con $1,000,000
    score_10m = sa.Column(sa.Float, nullable=False)  # Score con $10,000,000
    
    # Análisis del efecto de escala
    scale_sensitivity = sa.Column(sa.Float, nullable=False)  # Sensibilidad al cambio de escala (0-1)
    max_effective_capital = sa.Column(sa.Float, nullable=True)  # Capital máximo efectivo estimado
    
    # Factores limitantes
    limiting_factor = sa.Column(sa.String(50), nullable=True)  # Factor principal que limita la escala
    saturation_point = sa.Column(sa.Float, nullable=True)  # Punto de saturación de capital estimado
    
    def __repr__(self):
        return f"<CapitalScaleEffect id={self.id} sensitivity={self.scale_sensitivity:.2f}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "cryptocurrency_id": self.cryptocurrency_id,
            "analysis_date": self.analysis_date.isoformat() if self.analysis_date else None,
            "score_10k": self.score_10k,
            "score_100k": self.score_100k,
            "score_1m": self.score_1m,
            "score_10m": self.score_10m,
            "scale_sensitivity": self.scale_sensitivity,
            "max_effective_capital": self.max_effective_capital,
            "limiting_factor": self.limiting_factor,
            "saturation_point": self.saturation_point
        }