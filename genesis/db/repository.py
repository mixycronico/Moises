"""
Database repository patterns for data access.

This module provides repository classes for database operations
following the repository pattern.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Type, TypeVar, Generic, Union
from datetime import datetime, timedelta

from sqlalchemy import create_engine, text, func, and_, or_, desc
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

from genesis.config.settings import settings
from genesis.db.models import Base, Exchange, Symbol, Candle, Strategy, Signal, Trade, Balance, PerformanceMetric
from genesis.utils.logger import setup_logging

# Type variable for models
T = TypeVar('T')


class DatabaseManager:
    """
    Database manager for connection and session handling.
    
    This class manages database connections and sessions, handling
    both synchronous and asynchronous database operations.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            connection_string: Database connection string (optional)
        """
        self.logger = setup_logging('database_manager')
        
        # Get connection string from settings if not provided
        if connection_string is None:
            connection_string = settings.get('database.connection_string')
        
        self.connection_string = connection_string
        
        # Create sync engine
        self.engine = create_engine(
            self.connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={}
        )
        
        # Create async engine if using asyncpg
        if 'postgresql://' in self.connection_string:
            async_connection_string = self.connection_string.replace('postgresql://', 'postgresql+asyncpg://')
            self.async_engine = create_async_engine(
                async_connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={}
            )
            self.async_session_factory = sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        else:
            self.async_engine = None
            self.async_session_factory = None
        
        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.session = scoped_session(self.session_factory)
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(self.engine)
            self.logger.info("Database tables created")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """
        Get a database session.
        
        Returns:
            Database session
        """
        return self.session()
    
    async def get_async_session(self) -> AsyncSession:
        """
        Get an async database session.
        
        Returns:
            Async database session
        """
        if not self.async_session_factory:
            raise RuntimeError("Async database not configured")
        
        return self.async_session_factory()
    
    def close(self) -> None:
        """Close database connections."""
        self.session.remove()
        self.engine.dispose()
        self.logger.info("Database connections closed")
    
    async def close_async(self) -> None:
        """Close async database connections."""
        if self.async_engine:
            await self.async_engine.dispose()
            self.logger.info("Async database connections closed")


class BaseRepository(Generic[T]):
    """
    Base repository for database operations.
    
    This generic class provides basic CRUD operations for database models.
    """
    
    def __init__(self, db_manager: DatabaseManager, model_class: Type[T]):
        """
        Initialize the repository.
        
        Args:
            db_manager: Database manager
            model_class: Model class
        """
        self.db_manager = db_manager
        self.model_class = model_class
        self.logger = setup_logging(f"{model_class.__name__.lower()}_repository")
    
    def get_by_id(self, id: int) -> Optional[T]:
        """
        Get a single entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity instance or None
        """
        session = self.db_manager.get_session()
        try:
            return session.query(self.model_class).filter(self.model_class.id == id).first()
        except Exception as e:
            self.logger.error(f"Error getting {self.model_class.__name__} by ID {id}: {e}")
            return None
        finally:
            session.close()
    
    def get_all(self) -> List[T]:
        """
        Get all entities.
        
        Returns:
            List of entity instances
        """
        session = self.db_manager.get_session()
        try:
            return session.query(self.model_class).all()
        except Exception as e:
            self.logger.error(f"Error getting all {self.model_class.__name__}: {e}")
            return []
        finally:
            session.close()
    
    def create(self, entity_data: Dict[str, Any]) -> Optional[T]:
        """
        Create a new entity.
        
        Args:
            entity_data: Entity data
            
        Returns:
            Created entity or None
        """
        session = self.db_manager.get_session()
        try:
            entity = self.model_class(**entity_data)
            session.add(entity)
            session.commit()
            return entity
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error creating {self.model_class.__name__}: {e}")
            return None
        finally:
            session.close()
    
    def update(self, id: int, entity_data: Dict[str, Any]) -> Optional[T]:
        """
        Update an entity.
        
        Args:
            id: Entity ID
            entity_data: Entity data to update
            
        Returns:
            Updated entity or None
        """
        session = self.db_manager.get_session()
        try:
            entity = session.query(self.model_class).filter(self.model_class.id == id).first()
            if entity:
                for key, value in entity_data.items():
                    setattr(entity, key, value)
                session.commit()
                return entity
            return None
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error updating {self.model_class.__name__} {id}: {e}")
            return None
        finally:
            session.close()
    
    def delete(self, id: int) -> bool:
        """
        Delete an entity.
        
        Args:
            id: Entity ID
            
        Returns:
            True if successful, False otherwise
        """
        session = self.db_manager.get_session()
        try:
            entity = session.query(self.model_class).filter(self.model_class.id == id).first()
            if entity:
                session.delete(entity)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error deleting {self.model_class.__name__} {id}: {e}")
            return False
        finally:
            session.close()
    
    async def get_by_id_async(self, id: int) -> Optional[T]:
        """
        Get a single entity by ID (async).
        
        Args:
            id: Entity ID
            
        Returns:
            Entity instance or None
        """
        async_session = await self.db_manager.get_async_session()
        try:
            query = select(self.model_class).where(self.model_class.id == id)
            result = await async_session.execute(query)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Error getting {self.model_class.__name__} by ID {id} async: {e}")
            return None
        finally:
            await async_session.close()
    
    async def create_async(self, entity_data: Dict[str, Any]) -> Optional[T]:
        """
        Create a new entity (async).
        
        Args:
            entity_data: Entity data
            
        Returns:
            Created entity or None
        """
        async_session = await self.db_manager.get_async_session()
        try:
            entity = self.model_class(**entity_data)
            async_session.add(entity)
            await async_session.commit()
            return entity
        except Exception as e:
            await async_session.rollback()
            self.logger.error(f"Error creating {self.model_class.__name__} async: {e}")
            return None
        finally:
            await async_session.close()


class TradeRepository(BaseRepository[Trade]):
    """
    Repository for trade operations.
    
    This class provides specialized methods for trade data access.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the trade repository."""
        super().__init__(db_manager, Trade)
    
    def get_by_trade_id(self, trade_id: str) -> Optional[Trade]:
        """
        Get a trade by its trade ID.
        
        Args:
            trade_id: Trade ID
            
        Returns:
            Trade or None
        """
        session = self.db_manager.get_session()
        try:
            return session.query(Trade).filter(Trade.trade_id == trade_id).first()
        except Exception as e:
            self.logger.error(f"Error getting trade by trade ID {trade_id}: {e}")
            return None
        finally:
            session.close()
    
    def get_open_trades(self, symbol_id: Optional[int] = None) -> List[Trade]:
        """
        Get all open trades.
        
        Args:
            symbol_id: Optional symbol ID filter
            
        Returns:
            List of open trades
        """
        session = self.db_manager.get_session()
        try:
            query = session.query(Trade).filter(Trade.status == 'open')
            if symbol_id:
                query = query.filter(Trade.symbol_id == symbol_id)
            return query.all()
        except Exception as e:
            self.logger.error(f"Error getting open trades: {e}")
            return []
        finally:
            session.close()
    
    def get_trades_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Trade]:
        """
        Get trades within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trades
        """
        session = self.db_manager.get_session()
        try:
            return session.query(Trade).filter(
                Trade.entry_time >= start_date,
                Trade.entry_time <= end_date
            ).all()
        except Exception as e:
            self.logger.error(f"Error getting trades by date range: {e}")
            return []
        finally:
            session.close()
    
    def close_trade(
        self, trade_id: str, exit_price: float, exit_time: datetime, 
        realized_pnl: float, fees: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Close a trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_time: Exit time
            realized_pnl: Realized profit/loss
            fees: Fees
            
        Returns:
            Updated trade or None
        """
        session = self.db_manager.get_session()
        try:
            trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            if trade:
                trade.status = 'closed'
                trade.exit_price = exit_price
                trade.exit_time = exit_time
                trade.realized_pnl = realized_pnl
                if fees is not None:
                    trade.fees = fees
                session.commit()
                return trade
            return None
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error closing trade {trade_id}: {e}")
            return None
        finally:
            session.close()


class CandleRepository(BaseRepository[Candle]):
    """
    Repository for candle operations.
    
    This class provides specialized methods for candle data access.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the candle repository."""
        super().__init__(db_manager, Candle)
    
    def get_candles(
        self, symbol_id: int, timeframe: str, limit: int = 100, 
        start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[Candle]:
        """
        Get candles for a symbol and timeframe.
        
        Args:
            symbol_id: Symbol ID
            timeframe: Timeframe
            limit: Maximum number of candles
            start_time: Start time
            end_time: End time
            
        Returns:
            List of candles
        """
        session = self.db_manager.get_session()
        try:
            query = session.query(Candle).filter(
                Candle.symbol_id == symbol_id,
                Candle.timeframe == timeframe
            )
            
            if start_time:
                query = query.filter(Candle.timestamp >= start_time)
            if end_time:
                query = query.filter(Candle.timestamp <= end_time)
            
            return query.order_by(Candle.timestamp.desc()).limit(limit).all()
        except Exception as e:
            self.logger.error(f"Error getting candles for symbol {symbol_id}, timeframe {timeframe}: {e}")
            return []
        finally:
            session.close()
    
    def save_candles(self, candles: List[Dict[str, Any]]) -> int:
        """
        Save multiple candles.
        
        Args:
            candles: List of candle data dictionaries
            
        Returns:
            Number of candles saved
        """
        session = self.db_manager.get_session()
        try:
            count = 0
            for candle_data in candles:
                try:
                    # Check for existing candle to avoid duplicates
                    existing = session.query(Candle).filter(
                        Candle.exchange_id == candle_data['exchange_id'],
                        Candle.symbol_id == candle_data['symbol_id'],
                        Candle.timestamp == candle_data['timestamp'],
                        Candle.timeframe == candle_data['timeframe']
                    ).first()
                    
                    if existing:
                        # Update existing candle
                        for key, value in candle_data.items():
                            if key not in ('id', 'created_at'):
                                setattr(existing, key, value)
                    else:
                        # Create new candle
                        candle = Candle(**candle_data)
                        session.add(candle)
                    
                    count += 1
                except Exception as e:
                    self.logger.error(f"Error saving individual candle: {e}")
            
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving candles: {e}")
            return 0
        finally:
            session.close()


class PerformanceRepository(BaseRepository[PerformanceMetric]):
    """
    Repository for performance metrics operations.
    
    This class provides specialized methods for performance data access.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the performance repository."""
        super().__init__(db_manager, PerformanceMetric)
    
    def get_performance_history(
        self, start_date: datetime, end_date: datetime
    ) -> List[PerformanceMetric]:
        """
        Get performance metrics history.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of performance metrics
        """
        session = self.db_manager.get_session()
        try:
            return session.query(PerformanceMetric).filter(
                PerformanceMetric.timestamp >= start_date,
                PerformanceMetric.timestamp <= end_date
            ).order_by(PerformanceMetric.timestamp).all()
        except Exception as e:
            self.logger.error(f"Error getting performance history: {e}")
            return []
        finally:
            session.close()
    
    def get_latest_performance(self) -> Optional[PerformanceMetric]:
        """
        Get the latest performance metric.
        
        Returns:
            Latest performance metric or None
        """
        session = self.db_manager.get_session()
        try:
            return session.query(PerformanceMetric).order_by(
                PerformanceMetric.timestamp.desc()
            ).first()
        except Exception as e:
            self.logger.error(f"Error getting latest performance: {e}")
            return None
        finally:
            session.close()

