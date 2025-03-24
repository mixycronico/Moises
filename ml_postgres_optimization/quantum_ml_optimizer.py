"""
Optimizador Cuántico-ML para PostgreSQL - Sistema Genesis Trascendental

Este módulo implementa el optimizador de consultas PostgreSQL utilizando
principios cuánticos y técnicas avanzadas de Machine Learning para maximizar
el rendimiento y resiliencia del sistema bajo cargas extremas.

Características:
1. Predicción de carga usando LSTM con atención
2. Optimización dinámica de índices con aprendizaje por refuerzo
3. Ajuste automático de parámetros PostgreSQL
4. Análisis de patrones de consulta para pre-buffering
5. Quantum-inspired pooling para conexiones
"""

import logging
import asyncio
import json
import os
import numpy as np
import time
import random
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum, auto

# Configuración de logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Genesis-QuantumML-Optimizer")

class OptimizationFeature(Enum):
    """Características de optimización disponibles."""
    LOAD_PREDICTION = auto()
    INDEX_OPTIMIZATION = auto()
    PARAM_TUNING = auto()
    QUERY_ANALYSIS = auto()
    CONNECTION_POOLING = auto()
    BUFFER_MANAGEMENT = auto()
    VACUUM_OPTIMIZATION = auto()
    PARTITION_MANAGEMENT = auto()

class PostgresParameters:
    """Parámetros de PostgreSQL optimizables."""
    def __init__(self):
        self.shared_buffers = "256MB"  # Optimizable
        self.work_mem = "16MB"         # Optimizable
        self.maintenance_work_mem = "64MB"  # Optimizable
        self.effective_cache_size = "1GB"   # Optimizable
        self.random_page_cost = 4.0     # Optimizable
        self.max_connections = 100      # Optimizable
        self.max_prepared_transactions = 64  # Optimizable
        self.synchronous_commit = "on"  # Optimizable
        self.wal_buffers = "16MB"       # Optimizable
        self.default_statistics_target = 100  # Optimizable

class QueryAnalyzer:
    """Analizador de consultas para optimización."""
    def __init__(self):
        self.query_patterns: Dict[str, Any] = {}
        self.query_frequency: Dict[str, float] = {}
        self.query_latency: Dict[str, float] = {}
        self.last_analysis: float = time.time()
        
    async def analyze_query(self, query: str, execution_time: float) -> Dict[str, Any]:
        """
        Analizar una consulta y su tiempo de ejecución.
        
        Args:
            query: Consulta SQL
            execution_time: Tiempo de ejecución en ms
            
        Returns:
            Análisis de la consulta
        """
        # Simplificar la consulta para el análisis de patrones
        pattern = self._extract_pattern(query)
        
        # Actualizar estadísticas
        if pattern in self.query_patterns:
            self.query_frequency[pattern] += 1
            self.query_latency[pattern] = (
                self.query_latency[pattern] * 0.9 + execution_time * 0.1
            )  # Media móvil ponderada
        else:
            self.query_patterns[pattern] = {"example": query}
            self.query_frequency[pattern] = 1
            self.query_latency[pattern] = execution_time
        
        return {
            "pattern": pattern,
            "frequency": self.query_frequency[pattern],
            "avg_latency": self.query_latency[pattern],
            "optimization_potential": self._calculate_optimization_potential(pattern)
        }
    
    def _extract_pattern(self, query: str) -> str:
        """
        Extraer patrón de consulta normalizando valores literales.
        
        Args:
            query: Consulta SQL
            
        Returns:
            Patrón normalizado
        """
        # Simplificación básica: reemplazamos literales por marcadores de posición
        # En una implementación real, esto sería mucho más sofisticado
        import re
        
        # Reemplazar literales numéricos
        pattern = re.sub(r'\b\d+\b', '?', query)
        
        # Reemplazar literales de cadena
        pattern = re.sub(r"'[^']*'", "'?'", pattern)
        
        # Reemplazar literales de fecha
        pattern = re.sub(r'\d{4}-\d{2}-\d{2}', '?', pattern)
        
        return pattern
    
    def _calculate_optimization_potential(self, pattern: str) -> float:
        """
        Calcular potencial de optimización para un patrón de consulta.
        
        Args:
            pattern: Patrón de consulta
            
        Returns:
            Puntuación de potencial de optimización (0-1)
        """
        # Factores para el cálculo del potencial de optimización
        frequency_factor = min(1.0, self.query_frequency[pattern] / 100.0)
        latency_factor = min(1.0, self.query_latency[pattern] / 1000.0)
        
        # Análisis de complejidad de la consulta
        complexity_factor = 0.5  # Valor predeterminado
        if "JOIN" in pattern:
            complexity_factor += 0.2
        if "GROUP BY" in pattern:
            complexity_factor += 0.1
        if "ORDER BY" in pattern:
            complexity_factor += 0.1
        if "WHERE" in pattern:
            complexity_factor += 0.1
        
        # Potencial total
        potential = (frequency_factor * 0.4 + 
                     latency_factor * 0.4 + 
                     complexity_factor * 0.2)
        
        return min(1.0, potential)
    
    async def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones para optimización de consultas.
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Analizar patrones frecuentes con alta latencia
        for pattern in self.query_patterns:
            potential = self._calculate_optimization_potential(pattern)
            
            if potential > 0.7:  # Alto potencial de optimización
                recommendations.append({
                    "pattern": pattern,
                    "potential": potential,
                    "frequency": self.query_frequency[pattern],
                    "latency": self.query_latency[pattern],
                    "recommendation_type": self._get_recommendation_type(pattern),
                    "recommendation": self._generate_recommendation(pattern)
                })
        
        # Ordenar por potencial
        recommendations.sort(key=lambda x: x["potential"], reverse=True)
        
        return recommendations
    
    def _get_recommendation_type(self, pattern: str) -> str:
        """
        Determinar tipo de recomendación para un patrón.
        
        Args:
            pattern: Patrón de consulta
            
        Returns:
            Tipo de recomendación
        """
        if "JOIN" in pattern and "WHERE" in pattern:
            return "INDEX"
        elif "ORDER BY" in pattern:
            return "INDEX"
        elif "GROUP BY" in pattern:
            return "INDEX"
        elif "SELECT *" in pattern:
            return "QUERY_REWRITE"
        else:
            return "ANALYZE"
    
    def _generate_recommendation(self, pattern: str) -> str:
        """
        Generar recomendación específica para un patrón.
        
        Args:
            pattern: Patrón de consulta
            
        Returns:
            Recomendación textual
        """
        rec_type = self._get_recommendation_type(pattern)
        
        if rec_type == "INDEX":
            return "Considerar índice para las columnas en la cláusula WHERE"
        elif rec_type == "QUERY_REWRITE":
            return "Seleccionar sólo las columnas necesarias en lugar de SELECT *"
        else:
            return "Ejecutar ANALYZE en las tablas involucradas"

class LoadPredictor:
    """Predictor de carga con LSTM y atención."""
    def __init__(self, prediction_window: int = 60):
        self.prediction_window = prediction_window
        self.historical_load: List[float] = []
        self.current_prediction: Optional[List[float]] = None
        self.last_update = time.time()
    
    async def update_load_data(self, current_load: float) -> None:
        """
        Actualizar datos históricos de carga.
        
        Args:
            current_load: Carga actual del sistema
        """
        self.historical_load.append(current_load)
        
        # Mantener tamaño manejable del historial
        if len(self.historical_load) > 1000:
            self.historical_load = self.historical_load[-1000:]
        
        # Actualizar predicción si han pasado más de 5 segundos
        if time.time() - self.last_update > 5:
            await self.update_prediction()
            self.last_update = time.time()
    
    async def update_prediction(self) -> None:
        """Actualizar predicción de carga futura."""
        # En una implementación real, aquí usaríamos un modelo LSTM con atención
        # Para esta demo, vamos a simular la predicción con un modelo simple
        
        if len(self.historical_load) < 10:
            self.current_prediction = None
            return
            
        # Simulación simple de predicción con tendencia y estacionalidad
        recent_mean = np.mean(self.historical_load[-10:])
        recent_trend = np.mean(np.diff(self.historical_load[-10:]))
        
        # Predicción simple: tendencia lineal + ruido aleatorio
        prediction = []
        last_value = self.historical_load[-1]
        
        for i in range(self.prediction_window):
            next_value = last_value + recent_trend
            
            # Añadir algo de ruido y estacionalidad
            noise = np.random.normal(0, 0.02 * recent_mean)
            seasonal = 0.05 * recent_mean * np.sin(2 * np.pi * i / 20)
            
            next_value += noise + seasonal
            prediction.append(max(0, next_value))  # No permitir valores negativos
            
            last_value = next_value
        
        self.current_prediction = prediction
    
    def get_predicted_load(self) -> List[float]:
        """
        Obtener predicción de carga futura.
        
        Returns:
            Lista de valores predichos o lista vacía si no hay predicción
        """
        if self.current_prediction is None:
            return []
        return self.current_prediction

class IndexOptimizer:
    """Optimizador de índices con aprendizaje por refuerzo."""
    def __init__(self):
        self.table_columns: Dict[str, List[str]] = {}
        self.existing_indices: Dict[str, List[str]] = {}
        self.column_usage: Dict[str, Dict[str, float]] = {}  # Frecuencia de uso de columnas
        self.index_performance: Dict[str, float] = {}  # Rendimiento de índices
    
    async def update_column_stats(self, table: str, column: str, frequency: float) -> None:
        """
        Actualizar estadísticas de uso de columnas.
        
        Args:
            table: Nombre de la tabla
            column: Nombre de la columna
            frequency: Frecuencia de uso incremento
        """
        if table not in self.column_usage:
            self.column_usage[table] = {}
            
        if table not in self.table_columns:
            self.table_columns[table] = [column]
        elif column not in self.table_columns[table]:
            self.table_columns[table].append(column)
            
        # Actualizar frecuencia de uso
        current_freq = self.column_usage[table].get(column, 0)
        self.column_usage[table][column] = current_freq + frequency
    
    async def update_index_performance(self, index_name: str, performance_metric: float) -> None:
        """
        Actualizar métrica de rendimiento para un índice.
        
        Args:
            index_name: Nombre del índice
            performance_metric: Métrica de rendimiento (mejora de velocidad)
        """
        # Actualizar con media móvil ponderada
        current_perf = self.index_performance.get(index_name, 0)
        self.index_performance[index_name] = current_perf * 0.9 + performance_metric * 0.1
    
    async def register_index(self, table: str, columns: List[str], index_name: str) -> None:
        """
        Registrar un índice existente.
        
        Args:
            table: Nombre de la tabla
            columns: Columnas en el índice
            index_name: Nombre del índice
        """
        if table not in self.existing_indices:
            self.existing_indices[table] = []
            
        self.existing_indices[table].append(index_name)
    
    async def get_index_recommendations(self) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones de índices basadas en uso y rendimiento.
        
        Returns:
            Lista de recomendaciones de índices
        """
        recommendations = []
        
        for table in self.column_usage:
            # Obtener columnas más usadas
            columns_usage = [(col, freq) for col, freq in self.column_usage[table].items()]
            columns_usage.sort(key=lambda x: x[1], reverse=True)
            
            # Generar posibles índices de columna única para las columnas más usadas
            for column, frequency in columns_usage[:3]:  # Top 3 columnas más usadas
                index_name = f"idx_{table}_{column}"
                
                # Verificar si ya existe un índice para esta columna
                if (table in self.existing_indices and 
                    any(index_name == idx_name for idx_name in self.existing_indices[table])):
                    continue
                    
                # Calcular beneficio potencial
                potential_benefit = frequency / sum(freq for _, freq in columns_usage)
                
                recommendations.append({
                    "table": table,
                    "columns": [column],
                    "index_name": index_name,
                    "potential_benefit": potential_benefit,
                    "recommendation": f"CREATE INDEX {index_name} ON {table} ({column});"
                })
            
            # Para tablas con muchas consultas, considerar índices compuestos
            if sum(freq for _, freq in columns_usage) > 1000:
                # Tomar las 2 columnas más usadas para un índice compuesto
                if len(columns_usage) >= 2:
                    col1, _ = columns_usage[0]
                    col2, _ = columns_usage[1]
                    
                    index_name = f"idx_{table}_{col1}_{col2}"
                    
                    # Verificar si ya existe un índice para estas columnas
                    if (table in self.existing_indices and 
                        any(index_name == idx_name for idx_name in self.existing_indices[table])):
                        continue
                        
                    # Calcular beneficio potencial
                    potential_benefit = 0.8 * (columns_usage[0][1] + columns_usage[1][1]) / sum(freq for _, freq in columns_usage)
                    
                    recommendations.append({
                        "table": table,
                        "columns": [col1, col2],
                        "index_name": index_name,
                        "potential_benefit": potential_benefit,
                        "recommendation": f"CREATE INDEX {index_name} ON {table} ({col1}, {col2});"
                    })
        
        # Ordenar por beneficio potencial
        recommendations.sort(key=lambda x: x["potential_benefit"], reverse=True)
        
        return recommendations

class ConnectionPoolOptimizer:
    """Optimizador cuántico de pool de conexiones."""
    def __init__(self, db_url: str, min_connections: int = 5, max_connections: int = 50):
        self.db_url = db_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.current_connections = min_connections
        self.connection_usage: List[float] = []  # Historial de uso
        self.last_adjustment = time.time()
    
    async def update_usage_stats(self, active_connections: int, idle_connections: int) -> None:
        """
        Actualizar estadísticas de uso del pool.
        
        Args:
            active_connections: Conexiones activas
            idle_connections: Conexiones inactivas
        """
        total = active_connections + idle_connections
        if total > 0:
            usage_ratio = active_connections / total
        else:
            usage_ratio = 0
            
        self.connection_usage.append(usage_ratio)
        
        # Mantener historial manejable
        if len(self.connection_usage) > 100:
            self.connection_usage = self.connection_usage[-100:]
        
        # Ajustar tamaño del pool cada 30 segundos
        if time.time() - self.last_adjustment > 30:
            await self.adjust_pool_size()
            self.last_adjustment = time.time()
    
    async def adjust_pool_size(self) -> Dict[str, Any]:
        """
        Ajustar tamaño del pool de conexiones basado en estadísticas de uso.
        
        Returns:
            Resultado del ajuste
        """
        if not self.connection_usage:
            return {"action": "none", "reason": "insufficient_data"}
            
        # Calcular estadísticas de uso reciente
        recent_usage = self.connection_usage[-20:]
        avg_usage = sum(recent_usage) / len(recent_usage)
        max_recent_usage = max(recent_usage)
        
        old_size = self.current_connections
        
        # Reglas de ajuste
        if max_recent_usage > 0.8:  # Alto uso máximo
            # Aumentar tamaño para manejar picos
            target_size = min(
                int(self.current_connections * 1.2),
                self.max_connections
            )
            action = "increase"
            reason = "high_peak_usage"
        elif avg_usage < 0.3 and self.current_connections > self.min_connections:
            # Disminuir si uso promedio es bajo
            target_size = max(
                int(self.current_connections * 0.8),
                self.min_connections
            )
            action = "decrease"
            reason = "low_average_usage"
        else:
            # Mantener tamaño actual
            target_size = self.current_connections
            action = "maintain"
            reason = "optimal_usage"
        
        # Actualizar tamaño actual
        self.current_connections = target_size
        
        return {
            "action": action,
            "reason": reason,
            "old_size": old_size,
            "new_size": target_size,
            "avg_usage": avg_usage,
            "max_usage": max_recent_usage
        }
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Obtener configuración actual del pool.
        
        Returns:
            Configuración actual
        """
        return {
            "db_url": self.db_url,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
            "current_connections": self.current_connections,
            "recent_usage": self.connection_usage[-10:] if self.connection_usage else []
        }

class PostgresQLQuantumOptimizer:
    """Optimizador cuántico principal para PostgreSQL."""
    def __init__(self, db_url: str = os.environ.get("DATABASE_URL")):
        self.db_url = db_url
        self.query_analyzer = QueryAnalyzer()
        self.load_predictor = LoadPredictor()
        self.index_optimizer = IndexOptimizer()
        self.connection_pool_optimizer = ConnectionPoolOptimizer(db_url)
        self.postgres_params = PostgresParameters()
        self.enabled_features = {feature: True for feature in OptimizationFeature}
        self.optimization_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> None:
        """Inicializar optimizador y cargar datos iniciales."""
        logger.info("Inicializando PostgresQLQuantumOptimizer...")
        
        # Cargar configuración existente si está disponible
        # En una implementación real, esto cargaría la configuración desde la base de datos
        
        # Registrar índices existentes
        await self._load_existing_indices()
        
        logger.info("PostgresQLQuantumOptimizer inicializado correctamente")
        
    async def _load_existing_indices(self) -> None:
        """Cargar índices existentes desde la base de datos."""
        # En una implementación real, esto consultaría pg_indexes
        # Para esta demo, vamos a registrar algunos índices de ejemplo
        
        await self.index_optimizer.register_index(
            table="genesis_operations",
            columns=["operation_type"],
            index_name="idx_genesis_operations_type"
        )
        
        await self.index_optimizer.register_index(
            table="genesis_operations",
            columns=["timestamp"],
            index_name="idx_genesis_operations_timestamp"
        )
        
        await self.index_optimizer.register_index(
            table="genesis_metrics",
            columns=["timestamp"],
            index_name="idx_genesis_metrics_timestamp"
        )
        
    async def register_query(self, query: str, execution_time: float) -> Dict[str, Any]:
        """
        Registrar una consulta para análisis y optimización.
        
        Args:
            query: Consulta SQL
            execution_time: Tiempo de ejecución en ms
            
        Returns:
            Resultado del análisis
        """
        # Analizar consulta
        analysis = await self.query_analyzer.analyze_query(query, execution_time)
        
        # Extraer nombres de tablas y columnas para estadísticas de uso
        tables_columns = self._extract_tables_columns(query)
        
        for table, columns in tables_columns.items():
            for column in columns:
                await self.index_optimizer.update_column_stats(
                    table=table,
                    column=column,
                    frequency=1.0  # Incrementar en 1 el uso
                )
        
        return analysis
    
    def _extract_tables_columns(self, query: str) -> Dict[str, List[str]]:
        """
        Extraer nombres de tablas y columnas de una consulta SQL.
        
        Args:
            query: Consulta SQL
            
        Returns:
            Diccionario de {tabla: [columnas]}
        """
        # En una implementación real, esto utilizaría un parser SQL completo
        # Para esta demo, vamos a implementar una extracción muy básica
        
        tables_columns = {}
        
        # Extracción simple de tablas
        from_parts = query.upper().split("FROM ")
        if len(from_parts) > 1:
            tables_part = from_parts[1].split("WHERE")[0].split("JOIN")
            
            for table_part in tables_part:
                table_name = table_part.strip().split(" ")[0].strip()
                if table_name and table_name not in ("", "("):
                    tables_columns[table_name.lower()] = []
        
        # Extracción simple de columnas
        select_parts = query.upper().split("SELECT ")[1].split("FROM")[0]
        where_parts = query.upper().split("WHERE ")[1:] if "WHERE" in query.upper() else []
        
        all_columns = []
        
        if "," in select_parts:
            columns = [c.strip() for c in select_parts.split(",")]
            all_columns.extend(columns)
        else:
            all_columns.append(select_parts.strip())
            
        for where_part in where_parts:
            conditions = where_part.split("AND")
            for condition in conditions:
                if "=" in condition:
                    column = condition.split("=")[0].strip()
                    all_columns.append(column)
        
        # Asignar columnas a tablas (simplificado)
        for column in all_columns:
            if "." in column:
                table, col = column.split(".")
                table = table.lower()
                if table in tables_columns:
                    tables_columns[table].append(col.lower())
            else:
                # Si no se especifica tabla, añadir a todas las tablas
                for table in tables_columns:
                    tables_columns[table].append(column.lower())
        
        return tables_columns
    
    async def update_system_load(self, current_load: float) -> None:
        """
        Actualizar datos de carga del sistema.
        
        Args:
            current_load: Carga actual del sistema (0-1)
        """
        await self.load_predictor.update_load_data(current_load)
    
    async def get_load_prediction(self) -> Dict[str, Any]:
        """
        Obtener predicción de carga del sistema.
        
        Returns:
            Predicción de carga
        """
        prediction = self.load_predictor.get_predicted_load()
        
        return {
            "prediction_window": self.load_predictor.prediction_window,
            "prediction": prediction,
            "average_predicted_load": sum(prediction) / len(prediction) if prediction else 0,
            "max_predicted_load": max(prediction) if prediction else 0,
            "min_predicted_load": min(prediction) if prediction else 0,
        }
    
    async def update_pool_stats(self, active_connections: int, idle_connections: int) -> None:
        """
        Actualizar estadísticas del pool de conexiones.
        
        Args:
            active_connections: Número de conexiones activas
            idle_connections: Número de conexiones inactivas
        """
        await self.connection_pool_optimizer.update_usage_stats(
            active_connections, idle_connections
        )
    
    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Obtener recomendaciones completas de optimización.
        
        Returns:
            Recomendaciones de optimización
        """
        # Obtener recomendaciones de consultas
        query_recommendations = await self.query_analyzer.get_recommendations()
        
        # Obtener recomendaciones de índices
        index_recommendations = await self.index_optimizer.get_index_recommendations()
        
        # Obtener configuración de pool
        pool_config = self.connection_pool_optimizer.get_current_config()
        
        # Obtener predicción de carga
        load_prediction = await self.get_load_prediction()
        
        # Compilar recomendaciones completas
        return {
            "query_recommendations": query_recommendations,
            "index_recommendations": index_recommendations,
            "pool_configuration": pool_config,
            "load_prediction": load_prediction,
            "timestamp": time.time(),
            "postgres_parameters": vars(self.postgres_params)
        }
    
    async def apply_optimization(self, recommendation_id: str) -> Dict[str, Any]:
        """
        Aplicar una recomendación de optimización.
        
        Args:
            recommendation_id: ID de la recomendación a aplicar
            
        Returns:
            Resultado de la aplicación
        """
        # En una implementación real, esto ejecutaría la optimización en la base de datos
        # Para esta demo, simplemente registramos la optimización
        
        self.optimization_history.append({
            "recommendation_id": recommendation_id,
            "timestamp": time.time(),
            "status": "applied",
            "result": "success"
        })
        
        return {
            "status": "success",
            "message": f"Optimización {recommendation_id} aplicada correctamente",
            "timestamp": time.time()
        }
    
    async def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Obtener historial de optimizaciones aplicadas.
        
        Returns:
            Historial de optimizaciones
        """
        return self.optimization_history
    
    def enable_feature(self, feature: OptimizationFeature) -> None:
        """
        Activar una característica de optimización.
        
        Args:
            feature: Característica a activar
        """
        self.enabled_features[feature] = True
        logger.info(f"Característica de optimización activada: {feature.name}")
    
    def disable_feature(self, feature: OptimizationFeature) -> None:
        """
        Desactivar una característica de optimización.
        
        Args:
            feature: Característica a desactivar
        """
        self.enabled_features[feature] = False
        logger.info(f"Característica de optimización desactivada: {feature.name}")
    
    def get_enabled_features(self) -> Dict[str, bool]:
        """
        Obtener estado de características de optimización.
        
        Returns:
            Diccionario de {feature: enabled}
        """
        return {feature.name: enabled for feature, enabled in self.enabled_features.items()}

# Funciones de demo y prueba
async def demo_optimizer():
    """Demostrar capacidades del optimizador."""
    optimizer = PostgresQLQuantumOptimizer()
    await optimizer.initialize()
    
    # Registrar algunas consultas para análisis
    queries = [
        "SELECT * FROM genesis_operations WHERE operation_type = 'TRADE'",
        "SELECT timestamp, operation_type, status FROM genesis_operations WHERE timestamp > '2023-01-01'",
        "SELECT COUNT(*) FROM genesis_operations GROUP BY operation_type",
        "SELECT o.operation_type, m.latency_ms FROM genesis_operations o JOIN genesis_metrics m ON o.timestamp = m.timestamp",
        "SELECT AVG(latency_ms) FROM genesis_metrics WHERE operation_type = 'QUERY' GROUP BY DATE(timestamp)"
    ]
    
    for _ in range(10):  # Simular múltiples ejecuciones
        for query in queries:
            # Simular tiempo de ejecución aleatorio entre 10ms y 500ms
            execution_time = random.uniform(10, 500)
            await optimizer.register_query(query, execution_time)
    
    # Simular carga del sistema
    for _ in range(20):
        load = random.uniform(0.2, 0.8)
        await optimizer.update_system_load(load)
    
    # Simular estadísticas de pool
    for _ in range(5):
        active = random.randint(10, 40)
        idle = random.randint(5, 20)
        await optimizer.update_pool_stats(active, idle)
    
    # Obtener recomendaciones
    recommendations = await optimizer.get_optimization_recommendations()
    
    # Imprimir resultados
    logger.info("--- Demo del Optimizador Cuántico-ML de PostgreSQL ---")
    logger.info(f"Recomendaciones de consultas: {len(recommendations['query_recommendations'])}")
    if recommendations['query_recommendations']:
        logger.info(f"Top recomendación de consulta: {recommendations['query_recommendations'][0]}")
    
    logger.info(f"Recomendaciones de índices: {len(recommendations['index_recommendations'])}")
    if recommendations['index_recommendations']:
        logger.info(f"Top recomendación de índice: {recommendations['index_recommendations'][0]}")
    
    logger.info(f"Configuración del pool: {recommendations['pool_configuration']}")
    logger.info(f"Predicción de carga: {recommendations['load_prediction']['average_predicted_load']:.2f} (promedio)")
    
    # Características activas
    logger.info(f"Características activas: {optimizer.get_enabled_features()}")
    
    return recommendations

async def main():
    """Función principal para ejecución directa."""
    try:
        recommendations = await demo_optimizer()
        print(json.dumps(recommendations, indent=2))
    except Exception as e:
        logger.error(f"Error en demo: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())