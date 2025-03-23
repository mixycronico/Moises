from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, ForeignKey, JSON, Index, func
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property
from genesis.db.base import Base

class CryptoMetrics(Base):
    """Métricas en tiempo real de criptomonedas para clasificación trascendental."""
    __tablename__ = "crypto_metrics"
    __table_args__ = (
        Index('ix_crypto_metrics_symbol_timestamp', 'symbol', 'timestamp'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)

    # Métricas de volumen optimizadas
    volume_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume_usd_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume_change_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Métricas de precio con precisión divina
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_change_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_change_7d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Métricas de mercado
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_cap_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Métricas técnicas avanzadas
    rsi_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema_fast: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema_slow: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    adx: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    atr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    atr_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stoch_rsi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_width: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Métricas de exchange optimizadas
    spread_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exchange_count: Mapped[int] = mapped_column(Integer, default=0)

    # Métricas Fibonacci y niveles críticos
    fib_382_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fib_618_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    near_support: Mapped[bool] = mapped_column(Boolean, default=False)
    near_resistance: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relaciones trascendentales
    scores: Mapped["CryptoScores"] = relationship("CryptoScores", back_populates="metrics", uselist=False)
    predictions: Mapped[List["CryptoPredictions"]] = relationship("CryptoPredictions", back_populates="metrics")
    liquidity: Mapped[List["LiquidityData"]] = relationship("LiquidityData", back_populates="metrics")
    social_trends: Mapped[List["SocialTrends"]] = relationship("SocialTrends", back_populates="metrics")

    @hybrid_property
    def volatility(self) -> Optional[float]:
        """Propiedad híbrida para calcular volatilidad en tiempo real."""
        if self.price and self.atr:
            return self.atr / self.price
        return None

class CryptoScores(Base):
    """Puntuaciones calculadas con precisión divina para clasificación."""
    __tablename__ = "crypto_scores"
    __table_args__ = (
        Index('ix_crypto_scores_symbol_total_score', 'symbol', 'total_score'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metrics_id: Mapped[int] = mapped_column(Integer, ForeignKey("crypto_metrics.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Puntuaciones individuales normalizadas (0-1)
    volume_score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    change_score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    market_cap_score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    spread_score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    adoption_score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)

    # Puntuación total ponderada
    total_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)

    # Flags y parámetros de trading divino
    is_hot: Mapped[bool] = mapped_column(Boolean, default=False)
    allocation: Mapped[float] = mapped_column(Float, default=0.0)  # % de capital asignado
    drawdown_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit_multiplier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relación
    metrics: Mapped["CryptoMetrics"] = relationship("CryptoMetrics", back_populates="scores")

    @hybrid_property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calcula la relación riesgo-beneficio dinámicamente."""
        if self.drawdown_threshold and self.take_profit_multiplier:
            return self.take_profit_multiplier / abs(self.drawdown_threshold)
        return None

class CryptoPredictions(Base):
    """Predicciones avanzadas con modelo LSTM trascendental."""
    __tablename__ = "crypto_predictions"
    __table_args__ = (
        Index('ix_crypto_predictions_symbol_timestamp', 'symbol', 'timestamp'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metrics_id: Mapped[int] = mapped_column(Integer, ForeignKey("crypto_metrics.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Predicciones a múltiples horizontes
    prediction_1h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_4h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Probabilidades direccionales
    prob_up: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_down: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Confianza y métricas de evaluación
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Error observado

    # Relación
    metrics: Mapped["CryptoMetrics"] = relationship("CryptoMetrics", back_populates="predictions")

    @hybrid_property
    def expected_move(self) -> Optional[float]:
        """Calcula el movimiento esperado ponderado por confianza."""
        if self.prediction_24h and self.confidence:
            return self.prediction_24h * self.confidence
        return None

class SocialTrends(Base):
    """Tendencias sociales optimizadas para análisis trascendental."""
    __tablename__ = "social_trends"
    __table_args__ = (
        Index('ix_social_trends_symbol_trending_score', 'symbol', 'trending_score'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metrics_id: Mapped[int] = mapped_column(Integer, ForeignKey("crypto_metrics.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)

    # Métricas sociales
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # -1 a 1
    mentions_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    tweet_volume_24h: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    reddit_posts_24h: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    github_activity_7d: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    # Sentimiento desglosado
    sentiment_twitter: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_reddit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_news: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Puntuación trending
    trending_score: Mapped[Optional[float]] = mapped_column(Float, index=True, default=0.0)

    # Relación
    metrics: Mapped["CryptoMetrics"] = relationship("CryptoMetrics", back_populates="social_trends")

    @hybrid_property
    def social_momentum(self) -> Optional[float]:
        """Calcula el impulso social combinado."""
        if self.tweet_volume_24h and self.reddit_posts_24h:
            return (self.tweet_volume_24h + self.reddit_posts_24h) * self.sentiment_score
        return None

class LiquidityData(Base):
    """Datos de liquidez con profundidad divina."""
    __tablename__ = "liquidity_data"
    __table_args__ = (
        Index('ix_liquidity_data_symbol_exchange', 'symbol', 'exchange'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metrics_id: Mapped[int] = mapped_column(Integer, ForeignKey("crypto_metrics.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    exchange: Mapped[str] = mapped_column(String(30), nullable=False, index=True)

    # Profundidad y liquidez
    bid_ask_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    orderbook_depth_bids: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    orderbook_depth_asks: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    liquidity_score: Mapped[Optional[float]] = mapped_column(Float, index=True, default=0.0)
    slippage_1000usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slippage_10000usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relación
    metrics: Mapped["CryptoMetrics"] = relationship("CryptoMetrics", back_populates="liquidity")

    @hybrid_property
    def total_depth(self) -> Optional[float]:
        """Calcula la profundidad total del libro de órdenes."""
        if self.orderbook_depth_bids and self.orderbook_depth_asks:
            return self.orderbook_depth_bids + self.orderbook_depth_asks
        return None

class ClassifierLogs(Base):
    """Registro divino de actividades del clasificador."""
    __tablename__ = "classifier_logs"
    __table_args__ = (
        Index('ix_classifier_logs_timestamp_action', 'timestamp', 'action'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    success: Mapped[bool] = mapped_column(Boolean, default=True)

    @hybrid_property
    def duration(self) -> Optional[float]:
        """Calcula la duración de la acción si está registrada en details."""
        if 'start_time' in self.details and 'end_time' in self.details:
            return self.details['end_time'] - self.details['start_time']
        return None