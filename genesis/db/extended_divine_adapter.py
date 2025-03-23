"""
Adaptador divino de base de datos extendido para casos de uso especializados.

Este módulo extiende el DivineDatabaseAdapter con funcionalidades adicionales
específicas para ciertos casos de uso, demostrando cómo se puede heredar
y ampliar el adaptador base.
"""
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set

from genesis.db.divine_database import DivineDatabaseAdapter, get_divine_db_adapter

# Configuración de logging
logger = logging.getLogger("genesis.db.extended_divine_adapter")

class AnalyticsDBAdapter(DivineDatabaseAdapter):
    """
    Adaptador especializado para operaciones analíticas.
    
    Esta clase extiende el adaptador divino con funcionalidades específicas
    para consultas analíticas, optimizando para lectura masiva de datos.
    """
    
    def __init__(self, db_url: Optional[str] = None, pool_size: int = 15):
        """
        Inicializar adaptador analítico.
        
        Args:
            db_url: URL de conexión a la base de datos
            pool_size: Tamaño del pool (mayor para analítica)
        """
        # Aumentar tamaño de cache para analítica
        super().__init__(db_url=db_url, pool_size=pool_size)
        self.cache.max_size = 5000  # Mayor cache para analítica
        self.cache.default_ttl = 7200  # TTL más largo (2 horas)
        
        # Tabla de consultas analíticas frecuentes
        self._query_patterns: Dict[str, str] = {}
        
        logger.info("AnalyticsDBAdapter inicializado con cache extendido")
    
    def register_query_pattern(self, pattern_name: str, query_template: str) -> None:
        """
        Registrar un patrón de consulta analítica.
        
        Args:
            pattern_name: Nombre del patrón
            query_template: Plantilla SQL con placeholders
        """
        self._query_patterns[pattern_name] = query_template
        logger.info(f"Patrón de consulta registrado: {pattern_name}")
    
    def fetch_analytics_sync(self, pattern_name: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Ejecutar consulta analítica pre-registrada de forma síncrona.
        
        Args:
            pattern_name: Nombre del patrón de consulta
            params: Parámetros para la consulta
            
        Returns:
            Resultados de la consulta
            
        Raises:
            ValueError: Si el patrón no está registrado
        """
        if pattern_name not in self._query_patterns:
            raise ValueError(f"Patrón de consulta no registrado: {pattern_name}")
        
        query = self._query_patterns[pattern_name]
        return self.fetch_all_sync(query, params, use_cache=True)
    
    async def fetch_analytics_async(self, pattern_name: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Ejecutar consulta analítica pre-registrada de forma asíncrona.
        
        Args:
            pattern_name: Nombre del patrón de consulta
            params: Parámetros para la consulta
            
        Returns:
            Resultados de la consulta
            
        Raises:
            ValueError: Si el patrón no está registrado
        """
        if pattern_name not in self._query_patterns:
            raise ValueError(f"Patrón de consulta no registrado: {pattern_name}")
        
        query = self._query_patterns[pattern_name]
        return await self.fetch_all_async(query, params, use_cache=True)
    
    def prefetch_patterns(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Precargar resultados de todos los patrones de consulta en cache.
        
        Args:
            params: Parámetros comunes para las consultas
        """
        start_time = time.time()
        loaded = 0
        
        for pattern_name, query in self._query_patterns.items():
            try:
                self.fetch_all_sync(query, params, use_cache=True)
                loaded += 1
            except Exception as e:
                logger.error(f"Error precargando patrón {pattern_name}: {e}")
        
        duration = time.time() - start_time
        logger.info(f"Precarga completada: {loaded}/{len(self._query_patterns)} patrones en {duration:.2f}s")

class TimeSeriesDBAdapter(DivineDatabaseAdapter):
    """
    Adaptador especializado para datos de series temporales.
    
    Esta clase extiende el adaptador divino con funcionalidades específicas
    para operaciones con series temporales, como datos OHLCV, métricas
    temporales y eventos secuenciales.
    """
    
    def __init__(self, db_url: Optional[str] = None, partition_by: str = "day"):
        """
        Inicializar adaptador para series temporales.
        
        Args:
            db_url: URL de conexión a la base de datos
            partition_by: Granularidad de particionamiento ('day', 'week', 'month')
        """
        super().__init__(db_url=db_url)
        self.partition_by = partition_by
        self._timeseries_tables: Set[str] = set()
        
        logger.info(f"TimeSeriesDBAdapter inicializado con partición por {partition_by}")
    
    def register_timeseries_table(self, table_name: str) -> None:
        """
        Registrar una tabla como serie temporal.
        
        Args:
            table_name: Nombre de la tabla
        """
        self._timeseries_tables.add(table_name)
        logger.info(f"Tabla de serie temporal registrada: {table_name}")
    
    def insert_timeseries_sync(self, table_name: str, data: Dict[str, Any]) -> int:
        """
        Insertar dato en tabla de serie temporal de forma síncrona.
        
        Args:
            table_name: Nombre de la tabla
            data: Datos a insertar
            
        Returns:
            ID del registro creado
            
        Raises:
            ValueError: Si la tabla no está registrada como serie temporal
        """
        if table_name not in self._timeseries_tables:
            raise ValueError(f"Tabla no registrada como serie temporal: {table_name}")
        
        # Verificar que tiene campo timestamp
        if 'timestamp' not in data:
            raise ValueError("Los datos deben incluir campo 'timestamp'")
        
        # Construir columnas y valores dinámicamente
        columns = ', '.join(data.keys())
        placeholders = ', '.join([f'%({k})s' for k in data.keys()])
        
        query = f"""
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders})
            RETURNING id
        """
        
        result = self.fetch_one_sync(query, data)
        if result and 'id' in result:
            return result['id']
        return 0
    
    async def insert_timeseries_async(self, table_name: str, data: Dict[str, Any]) -> int:
        """
        Insertar dato en tabla de serie temporal de forma asíncrona.
        
        Args:
            table_name: Nombre de la tabla
            data: Datos a insertar
            
        Returns:
            ID del registro creado
            
        Raises:
            ValueError: Si la tabla no está registrada como serie temporal
        """
        if table_name not in self._timeseries_tables:
            raise ValueError(f"Tabla no registrada como serie temporal: {table_name}")
        
        # Verificar que tiene campo timestamp
        if 'timestamp' not in data:
            raise ValueError("Los datos deben incluir campo 'timestamp'")
        
        # Construir columnas y valores dinámicamente
        columns = ', '.join(data.keys())
        placeholders = ', '.join([f'%({k})s' for k in data.keys()])
        
        query = f"""
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders})
            RETURNING id
        """
        
        result = await self.fetch_one_async(query, data)
        if result and 'id' in result:
            return result['id']
        return 0
    
    def get_timeseries_sync(
        self, 
        table_name: str, 
        start_timestamp: str, 
        end_timestamp: str,
        fields: Optional[List[str]] = None,
        limit: int = 1000,
        order_by: str = "timestamp ASC"
    ) -> List[Dict[str, Any]]:
        """
        Obtener datos de serie temporal de forma síncrona.
        
        Args:
            table_name: Nombre de la tabla
            start_timestamp: Timestamp de inicio
            end_timestamp: Timestamp de fin
            fields: Campos a seleccionar (None para todos)
            limit: Límite de registros
            order_by: Ordenamiento
            
        Returns:
            Lista de registros
            
        Raises:
            ValueError: Si la tabla no está registrada como serie temporal
        """
        if table_name not in self._timeseries_tables:
            raise ValueError(f"Tabla no registrada como serie temporal: {table_name}")
        
        # Construir consulta
        fields_str = '*'
        if fields:
            fields_str = ', '.join(fields)
        
        query = f"""
            SELECT {fields_str}
            FROM {table_name}
            WHERE timestamp >= %(start_timestamp)s
            AND timestamp <= %(end_timestamp)s
            ORDER BY {order_by}
            LIMIT %(limit)s
        """
        
        params = {
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'limit': limit
        }
        
        return self.fetch_all_sync(query, params)
    
    async def get_timeseries_async(
        self, 
        table_name: str, 
        start_timestamp: str, 
        end_timestamp: str,
        fields: Optional[List[str]] = None,
        limit: int = 1000,
        order_by: str = "timestamp ASC"
    ) -> List[Dict[str, Any]]:
        """
        Obtener datos de serie temporal de forma asíncrona.
        
        Args:
            table_name: Nombre de la tabla
            start_timestamp: Timestamp de inicio
            end_timestamp: Timestamp de fin
            fields: Campos a seleccionar (None para todos)
            limit: Límite de registros
            order_by: Ordenamiento
            
        Returns:
            Lista de registros
            
        Raises:
            ValueError: Si la tabla no está registrada como serie temporal
        """
        if table_name not in self._timeseries_tables:
            raise ValueError(f"Tabla no registrada como serie temporal: {table_name}")
        
        # Construir consulta
        fields_str = '*'
        if fields:
            fields_str = ', '.join(fields)
        
        query = f"""
            SELECT {fields_str}
            FROM {table_name}
            WHERE timestamp >= %(start_timestamp)s
            AND timestamp <= %(end_timestamp)s
            ORDER BY {order_by}
            LIMIT %(limit)s
        """
        
        params = {
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'limit': limit
        }
        
        return await self.fetch_all_async(query, params)

# Singleton global para analytics
_analytics_db_adapter = None

def get_analytics_db_adapter(db_url: Optional[str] = None) -> AnalyticsDBAdapter:
    """
    Obtener instancia global del adaptador analítico.
    
    Args:
        db_url: URL de conexión (opcional)
        
    Returns:
        Instancia de AnalyticsDBAdapter
    """
    global _analytics_db_adapter
    
    if _analytics_db_adapter is None:
        _analytics_db_adapter = AnalyticsDBAdapter(db_url)
    
    return _analytics_db_adapter

# Singleton global para timeseries
_timeseries_db_adapter = None

def get_timeseries_db_adapter(db_url: Optional[str] = None) -> TimeSeriesDBAdapter:
    """
    Obtener instancia global del adaptador para series temporales.
    
    Args:
        db_url: URL de conexión (opcional)
        
    Returns:
        Instancia de TimeSeriesDBAdapter
    """
    global _timeseries_db_adapter
    
    if _timeseries_db_adapter is None:
        _timeseries_db_adapter = TimeSeriesDBAdapter(db_url)
    
    return _timeseries_db_adapter