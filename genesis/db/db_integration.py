"""
Integración de TimescaleDB con el Sistema Genesis.

Este módulo conecta el adaptador de TimescaleDB con el resto del sistema,
proporcionando interfaces sencillas para que los componentes del sistema
trabajen con la base de datos sin preocuparse por los detalles de implementación.
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta

from .timescaledb_adapter import TimescaleDBManager

class DatabaseAdapter:
    """
    Adaptador de base de datos para interactuar con TimescaleDB.
    
    Expone métodos asíncronos para uso en el sistema Genesis y
    maneja la comunicación con el TimescaleDBManager que opera en
    un hilo separado.
    """
    
    def __init__(self, 
                 db_manager: Optional[TimescaleDBManager] = None,
                 dsn: Optional[str] = None,
                 initialize: bool = True):
        """
        Inicializar adaptador de base de datos.
        
        Args:
            db_manager: Instancia del gestor TimescaleDB
            dsn: Cadena de conexión (si no se proporciona db_manager)
            initialize: Si es True, inicializa automáticamente
        """
        self.logger = logging.getLogger(__name__)
        
        # Usar db_manager proporcionado o crear uno nuevo
        if db_manager:
            self.db_manager = db_manager
        else:
            self.db_manager = TimescaleDBManager(dsn=dsn)
        
        # Inicializar
        if initialize:
            self.init()
            
        self.logger.info("DatabaseAdapter inicializado")
    
    def init(self) -> bool:
        """
        Inicializar adaptador.
        
        Returns:
            True si se inicializó correctamente
        """
        # Iniciar manager
        self.db_manager.start()
        
        # Configurar hipertablas
        self.db_manager.setup_hypertables()
        
        return True
    
    def shutdown(self) -> bool:
        """
        Cerrar adaptador.
        
        Returns:
            True si se cerró correctamente
        """
        # Detener manager
        self.db_manager.stop()
        
        return True
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> bool:
        """
        Ejecutar query de forma asíncrona.
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            
        Returns:
            True si se ejecutó correctamente
        """
        # Crear future para esperar resultado
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Callback para resolver future
        def callback(result):
            loop.call_soon_threadsafe(future.set_result, result)
        
        # Callback para error
        def error_callback(error):
            loop.call_soon_threadsafe(future.set_exception, Exception(error))
        
        # Ejecutar en hilo separado
        self.db_manager.execute(query, params, callback=callback)
        
        try:
            # Esperar resultado
            result = await asyncio.wait_for(future, timeout=10.0)
            return result
        except asyncio.TimeoutError:
            self.logger.error("Timeout ejecutando query")
            return False
        except Exception as e:
            self.logger.error(f"Error ejecutando query: {str(e)}")
            return False
    
    async def fetch(self, 
                   query: str, 
                   params: Optional[tuple] = None, 
                   as_dict: bool = True) -> List[Any]:
        """
        Obtener datos de forma asíncrona.
        
        Args:
            query: Consulta SQL
            params: Parámetros para la consulta
            as_dict: Si es True, devuelve resultados como diccionarios
            
        Returns:
            Lista de resultados
        """
        # Crear future para esperar resultado
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Callback para resolver future
        def callback(results):
            loop.call_soon_threadsafe(future.set_result, results)
        
        # Callback para error
        def error_callback(error):
            loop.call_soon_threadsafe(future.set_exception, Exception(error))
        
        # Ejecutar en hilo separado
        self.db_manager.fetch(
            query, 
            params, 
            callback=callback,
            error_callback=error_callback,
            as_dict=as_dict
        )
        
        try:
            # Esperar resultado
            results = await asyncio.wait_for(future, timeout=10.0)
            return results
        except asyncio.TimeoutError:
            self.logger.error("Timeout ejecutando fetch")
            return []
        except Exception as e:
            self.logger.error(f"Error ejecutando fetch: {str(e)}")
            return []
    
    async def transaction(self, queries: List[Dict[str, Any]]) -> bool:
        """
        Ejecutar múltiples queries en una transacción asíncrona.
        
        Args:
            queries: Lista de diccionarios con queries y parámetros
            
        Returns:
            True si se ejecutó correctamente
        """
        # Crear future para esperar resultado
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Callback para resolver future
        def callback(result):
            loop.call_soon_threadsafe(future.set_result, result)
        
        # Callback para error
        def error_callback(error):
            loop.call_soon_threadsafe(future.set_exception, Exception(error))
        
        # Ejecutar en hilo separado
        self.db_manager.transaction(
            queries, 
            callback=callback,
            error_callback=error_callback
        )
        
        try:
            # Esperar resultado
            result = await asyncio.wait_for(future, timeout=15.0)
            return result
        except asyncio.TimeoutError:
            self.logger.error("Timeout ejecutando transacción")
            return False
        except Exception as e:
            self.logger.error(f"Error ejecutando transacción: {str(e)}")
            return False
    
    async def save_market_data(self, data: List[Dict[str, Any]], symbol: str) -> bool:
        """
        Guardar datos de mercado.
        
        Args:
            data: Lista de datos OHLCV
            symbol: Símbolo
            
        Returns:
            True si se guardó correctamente
        """
        # Ejecutar en hilo separado
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.db_manager.bulk_insert_market_data(data, symbol)
        )
        
        return result
    
    async def get_market_data(self, 
                            symbol: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener datos de mercado.
        
        Args:
            symbol: Símbolo
            start_time: Tiempo de inicio (opcional)
            end_time: Tiempo de fin (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de datos OHLCV
        """
        # Construir query
        query = """
            SELECT timestamp, symbol, open, high, low, close, volume
            FROM market_data
            WHERE symbol = %s
        """
        
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= %s"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        # Ejecutar query
        results = await self.fetch(query, tuple(params))
        
        return results
    
    async def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Guardar información de una operación de trading.
        
        Args:
            trade_data: Datos de la operación
            
        Returns:
            True si se guardó correctamente
        """
        # Verificar datos mínimos
        if 'symbol' not in trade_data:
            self.logger.error("Datos de trade incompletos: falta symbol")
            return False
        
        # Preparar query
        query = """
            INSERT INTO trades 
            (timestamp, symbol, side, price, amount, cost, fee, realized_pnl, strategy, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Valores por defecto y extraer datos
        timestamp = trade_data.get('timestamp', datetime.now())
        symbol = trade_data['symbol']
        side = trade_data.get('side', 'unknown')
        price = trade_data.get('price', 0.0)
        amount = trade_data.get('amount', 0.0)
        cost = trade_data.get('cost', price * amount)
        fee = trade_data.get('fee', 0.0)
        realized_pnl = trade_data.get('realized_pnl', 0.0)
        strategy = trade_data.get('strategy', 'unknown')
        status = trade_data.get('status', 'executed')
        
        # Parámetros
        params = (
            timestamp, symbol, side, price, amount, 
            cost, fee, realized_pnl, strategy, status
        )
        
        # Ejecutar query
        result = await self.execute(query, params)
        
        return result
    
    async def save_performance(self, performance_data: Dict[str, Any]) -> bool:
        """
        Guardar métricas de rendimiento.
        
        Args:
            performance_data: Datos de rendimiento
            
        Returns:
            True si se guardó correctamente
        """
        # Preparar query
        query = """
            INSERT INTO performance 
            (timestamp, equity, balance, drawdown, daily_return, sharpe_ratio, sortino_ratio, win_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Valores por defecto y extraer datos
        timestamp = performance_data.get('timestamp', datetime.now())
        equity = performance_data.get('equity', 0.0)
        balance = performance_data.get('balance', 0.0)
        drawdown = performance_data.get('drawdown', 0.0)
        daily_return = performance_data.get('daily_return', 0.0)
        sharpe_ratio = performance_data.get('sharpe_ratio', 0.0)
        sortino_ratio = performance_data.get('sortino_ratio', 0.0)
        win_rate = performance_data.get('win_rate', 0.0)
        
        # Parámetros
        params = (
            timestamp, equity, balance, drawdown, 
            daily_return, sharpe_ratio, sortino_ratio, win_rate
        )
        
        # Ejecutar query
        result = await self.execute(query, params)
        
        return result
    
    async def get_recent_trades(self, 
                              symbol: Optional[str] = None, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener operaciones recientes.
        
        Args:
            symbol: Símbolo (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de operaciones
        """
        # Construir query
        query = """
            SELECT timestamp, symbol, side, price, amount, cost, fee, realized_pnl, strategy, status
            FROM trades
        """
        
        params = []
        
        if symbol:
            query += " WHERE symbol = %s"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        # Ejecutar query
        results = await self.fetch(query, tuple(params) if params else None)
        
        return results
    
    async def get_performance_history(self, 
                                   days: int = 30, 
                                   interval: str = 'day') -> List[Dict[str, Any]]:
        """
        Obtener historial de rendimiento.
        
        Args:
            days: Número de días a obtener
            interval: Intervalo de tiempo ('hour', 'day', 'week')
            
        Returns:
            Lista de métricas de rendimiento
        """
        # Determinar función de tiempo según intervalo
        if interval == 'hour':
            time_bucket = "time_bucket('1 hour', timestamp)"
        elif interval == 'week':
            time_bucket = "time_bucket('1 week', timestamp)"
        else:  # 'day' por defecto
            time_bucket = "time_bucket('1 day', timestamp)"
        
        # Construir query
        query = f"""
            SELECT 
                {time_bucket} as period,
                MAX(equity) as equity,
                MAX(balance) as balance,
                MAX(drawdown) as drawdown,
                AVG(daily_return) as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(sortino_ratio) as avg_sortino,
                AVG(win_rate) as avg_win_rate
            FROM performance
            WHERE timestamp >= NOW() - INTERVAL '{days} days'
            GROUP BY period
            ORDER BY period DESC
        """
        
        # Ejecutar query
        results = await self.fetch(query)
        
        return results
    
    async def get_db_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la base de datos.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = self.db_manager.get_stats()
        
        # Añadir información adicional si está disponible
        try:
            # Contar registros en tablas principales
            market_data_count = await self.fetch(
                "SELECT COUNT(*) as count FROM market_data", 
                as_dict=True
            )
            trades_count = await self.fetch(
                "SELECT COUNT(*) as count FROM trades", 
                as_dict=True
            )
            performance_count = await self.fetch(
                "SELECT COUNT(*) as count FROM performance", 
                as_dict=True
            )
            
            # Añadir conteos
            stats['table_counts'] = {
                'market_data': market_data_count[0]['count'] if market_data_count else 0,
                'trades': trades_count[0]['count'] if trades_count else 0,
                'performance': performance_count[0]['count'] if performance_count else 0
            }
            
            # Tamaño de tablas
            size_query = """
                SELECT 
                    table_name,
                    pg_size_pretty(pg_relation_size(quote_ident(table_name))) as table_size
                FROM 
                    information_schema.tables
                WHERE 
                    table_schema = 'public'
                    AND table_name IN ('market_data', 'trades', 'performance')
            """
            
            size_results = await self.fetch(size_query, as_dict=True)
            stats['table_sizes'] = {row['table_name']: row['table_size'] for row in size_results}
            
        except Exception as e:
            self.logger.error(f"Error obteniendo estadísticas adicionales: {str(e)}")
        
        return stats
    
    def get_sync_manager(self) -> TimescaleDBManager:
        """
        Obtener el gestor de base de datos subyacente para operaciones síncronas.
        
        Returns:
            Instancia de TimescaleDBManager
        """
        return self.db_manager