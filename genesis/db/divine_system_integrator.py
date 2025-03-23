"""
Integrador del Sistema Divino con Machine Learning para Genesis.

Este módulo vincula el Sistema Divino de Tareas de Base de Datos que utiliza
una arquitectura híbrida Redis+RabbitMQ con capacidades de ML predictivo,
integrándolo con el sistema Genesis existente.

La integración permite:
- Procesamiento ultrarrápido y confiable de operaciones de base de datos
- Predicción y optimización automática de recursos
- Priorización inteligente de tareas basada en aprendizaje continuo
- Tolerancia a fallos extrema con recuperación automática
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

# Importar componentes divinos
from .divine_task_queue import (
    DivineTaskQueue, 
    DivineConfig,
    OperationMode,
    divine_task,
    critical_task,
    high_priority_task,
    divine_transaction
)

# Importar el sistema ML
from .divine_ml import divine_ml, DivineMachineLearning

# Importar componentes de base de Genesis
from .base import get_db_session, async_db_operation

# Configuración de logging
logger = logging.getLogger("genesis.db.divine_system_integrator")

class DivineSystem:
    """
    Sistema Divino integrado para Genesis con capacidades ML.
    
    Esta clase integra:
    - Sistema de Cola Divina (Redis + RabbitMQ)
    - Sistema ML predictivo y adaptativo
    - Sistemas base de datos existentes en Genesis
    
    Proporciona una interfaz unificada para todas las operaciones de base
    de datos con priorización inteligente y tolerancia a fallos extrema.
    """
    
    def __init__(self, config: Optional[DivineConfig] = None):
        """
        Inicializar el Sistema Divino integrado.
        
        Args:
            config: Configuración opcional
        """
        # Sistema principal
        self.divine_queue = DivineTaskQueue(config)
        
        # Sistema ML (compartido globalmente)
        self.ml_system = divine_ml
        
        # Vincular ML con cola
        self.divine_queue.ml_system = self.ml_system
        
        # Estado y métricas
        self.initialized = False
        self.start_time = None
        self.operations_processed = 0
        self.operations_failed = 0
        
        # Datos de operaciones para ML
        self.recent_operations = []
        self.max_recent_operations = 1000
        
        logger.info("Sistema Divino integrado creado")
    
    async def initialize(self):
        """Inicializar el sistema divino completo."""
        if self.initialized:
            logger.info("Sistema Divino ya inicializado")
            return
        
        # Inicializar cola divina
        if not self.divine_queue.initialized:
            await self.divine_queue.initialize()
        
        # Registrar tiempo de inicio
        self.start_time = datetime.now()
        self.initialized = True
        
        logger.info("Sistema Divino integrado inicializado completamente")
    
    async def start(self):
        """Iniciar el sistema divino."""
        if not self.initialized:
            await self.initialize()
        
        # Iniciar cola divina
        if not self.divine_queue.running:
            await self.divine_queue.start()
        
        logger.info("Sistema Divino integrado iniciado")
    
    async def stop(self):
        """Detener el sistema divino."""
        if self.divine_queue.running:
            await self.divine_queue.stop()
        
        logger.info("Sistema Divino integrado detenido")
    
    async def execute_divine_query(self, sql: str, params: Optional[List[Any]] = None, 
                                  priority: int = 5, critical: bool = False,
                                  operation_type: str = "query") -> Any:
        """
        Ejecutar consulta SQL a través del sistema divino.
        
        Args:
            sql: Consulta SQL
            params: Parámetros para la consulta
            priority: Prioridad (1-10)
            critical: Si la operación es crítica
            operation_type: Tipo de operación
            
        Returns:
            Resultado de la consulta
        """
        if not self.initialized:
            await self.initialize()
        
        # Determinar tipo y volumen para ML
        is_select = sql.strip().lower().startswith("select")
        is_transaction = not is_select and "begin" not in sql.lower()
        
        # Estimar volumen basado en tamaño de consulta y parámetros
        volume = len(sql)
        if params:
            volume += len(str(params))
        
        # Normalizar volumen a escala 0-1000
        volume = min(1000, volume)
        
        # Predecir prioridad si ML disponible y no se especificó manualmente
        if priority == 5 and hasattr(self.ml_system, 'predict_priority'):
            try:
                predicted_priority = self.ml_system.predict_priority(
                    critical=critical,
                    volume=volume,
                    transactional=is_transaction
                )
                priority = predicted_priority
                logger.debug(f"Prioridad predicha para consulta: {priority}")
            except Exception as e:
                logger.error(f"Error al predecir prioridad: {e}")
        
        # Encapsular consulta en una función asíncrona
        @divine_task(priority=priority)
        async def _execute_divine_sql():
            session = await get_db_session()
            try:
                # Ejecutar consulta
                if is_select:
                    result = await session.fetch(sql, *params if params else [])
                    return [dict(row) for row in result]
                else:
                    result = await session.execute(sql, *params if params else [])
                    await session.commit()
                    return result
            except Exception as e:
                # Hacer rollback si no es consulta SELECT
                if not is_select:
                    await session.rollback()
                raise
            finally:
                await session.close()
        
        # Ejecutar a través del sistema divino
        start_time = time.time()
        try:
            result = await _execute_divine_sql()
            
            # Registrar operación exitosa para ML
            elapsed = time.time() - start_time
            self._record_operation_success(priority, elapsed, volume, is_transaction, critical)
            
            return result
            
        except Exception as e:
            # Registrar error para ML
            elapsed = time.time() - start_time
            self._record_operation_failure(priority, elapsed, volume, is_transaction, critical)
            
            # Propagar error
            raise
    
    async def execute_divine_transaction(self, queries: List[Dict[str, Any]], 
                                       priority: int = 8, critical: bool = True) -> List[Any]:
        """
        Ejecutar múltiples consultas en una transacción divina.
        
        Args:
            queries: Lista de consultas, cada una como un diccionario con keys 'query' y 'params'
            priority: Prioridad (1-10)
            critical: Si la operación es crítica
            
        Returns:
            Resultados de las consultas
        """
        if not self.initialized:
            await self.initialize()
        
        # Estimar volumen basado en número de consultas
        volume = sum(len(q.get("query", "")) for q in queries)
        volume = min(1000, volume)
        
        # Predecir prioridad si ML disponible y no se especificó manualmente
        if priority == 8 and hasattr(self.ml_system, 'predict_priority'):
            try:
                predicted_priority = self.ml_system.predict_priority(
                    critical=critical,
                    volume=volume,
                    transactional=True
                )
                priority = predicted_priority
                logger.debug(f"Prioridad predicha para transacción: {priority}")
            except Exception as e:
                logger.error(f"Error al predecir prioridad: {e}")
        
        # Encapsular transacción en una función asíncrona
        @divine_task(priority=priority)
        async def _execute_divine_transaction():
            session = await get_db_session()
            try:
                # Iniciar transacción
                transaction = await session.begin()
                
                results = []
                for query_data in queries:
                    sql = query_data.get("query")
                    params = query_data.get("params", [])
                    
                    # Ejecutar consulta
                    if sql.lower().strip().startswith(("select", "with")):
                        result = await session.fetch(sql, *params)
                        results.append([dict(row) for row in result])
                    else:
                        result = await session.execute(sql, *params)
                        results.append(result)
                
                # Confirmar transacción
                await transaction.commit()
                return results
                
            except Exception as e:
                # Hacer rollback en caso de error
                if 'transaction' in locals():
                    await transaction.rollback()
                raise
                
            finally:
                # Cerrar sesión
                await session.close()
        
        # Ejecutar a través del sistema divino
        start_time = time.time()
        try:
            results = await _execute_divine_transaction()
            
            # Registrar operación exitosa para ML
            elapsed = time.time() - start_time
            self._record_operation_success(priority, elapsed, volume, True, critical)
            
            return results
            
        except Exception as e:
            # Registrar error para ML
            elapsed = time.time() - start_time
            self._record_operation_failure(priority, elapsed, volume, True, critical)
            
            # Propagar error
            raise
    
    def _record_operation_success(self, priority: int, elapsed: float, 
                                volume: float, is_transaction: bool, critical: bool):
        """Registrar operación exitosa para ML."""
        self.operations_processed += 1
        
        # Registrar para predicción de carga
        current_time = time.time()
        if hasattr(self.ml_system, 'record_operation'):
            try:
                # Convertir en una tasa de operaciones/segundo
                ops_rate = 1.0 / elapsed if elapsed > 0 else 1.0
                self.ml_system.record_operation(current_time, ops_rate, elapsed)
            except Exception as e:
                logger.error(f"Error al registrar operación para ML: {e}")
        
        # Registrar asignación de prioridad
        if hasattr(self.ml_system, 'record_priority_assignment'):
            try:
                self.ml_system.record_priority_assignment(
                    critical=critical,
                    volume=volume,
                    transactional=is_transaction,
                    priority=priority
                )
            except Exception as e:
                logger.error(f"Error al registrar prioridad para ML: {e}")
        
        # Guardar en historial reciente
        self.recent_operations.append({
            "timestamp": current_time,
            "success": True,
            "elapsed": elapsed,
            "priority": priority,
            "volume": volume,
            "transactional": is_transaction,
            "critical": critical
        })
        
        # Limitar tamaño del historial
        if len(self.recent_operations) > self.max_recent_operations:
            self.recent_operations = self.recent_operations[-self.max_recent_operations:]
    
    def _record_operation_failure(self, priority: int, elapsed: float, 
                                volume: float, is_transaction: bool, critical: bool):
        """Registrar operación fallida para ML."""
        self.operations_failed += 1
        
        # Guardar en historial reciente
        current_time = time.time()
        self.recent_operations.append({
            "timestamp": current_time,
            "success": False,
            "elapsed": elapsed,
            "priority": priority,
            "volume": volume,
            "transactional": is_transaction,
            "critical": critical
        })
        
        # Limitar tamaño del historial
        if len(self.recent_operations) > self.max_recent_operations:
            self.recent_operations = self.recent_operations[-self.max_recent_operations:]
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema divino.
        
        Returns:
            Estadísticas detalladas
        """
        stats = {
            "divine_system": {
                "initialized": self.initialized,
                "operations_processed": self.operations_processed,
                "operations_failed": self.operations_failed,
                "success_rate": (self.operations_processed / (self.operations_processed + self.operations_failed)) * 100 if (self.operations_processed + self.operations_failed) > 0 else 0,
            }
        }
        
        # Obtener estadísticas de la cola divina
        if self.divine_queue:
            try:
                queue_stats = await self.divine_queue.get_stats_async()
                stats["divine_queue"] = queue_stats
            except Exception as e:
                stats["divine_queue"] = {"error": str(e)}
        
        # Obtener estadísticas del sistema ML
        if self.ml_system:
            try:
                ml_stats = self.ml_system.get_stats()
                stats["divine_ml"] = ml_stats
            except Exception as e:
                stats["divine_ml"] = {"error": str(e)}
        
        # Calcular tiempo de ejecución
        if self.start_time:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            stats["divine_system"]["uptime_seconds"] = uptime_seconds
            stats["divine_system"]["start_time"] = self.start_time.isoformat()
        
        return stats

# Instancia global
divine_system = DivineSystem()

# Funciones de conveniencia para integración
async def execute_divine_sql(sql: str, params: Optional[List[Any]] = None, 
                           priority: int = 5, critical: bool = False) -> Any:
    """
    Ejecutar consulta SQL usando el sistema divino.
    
    Args:
        sql: Consulta SQL
        params: Parámetros para la consulta
        priority: Prioridad (1-10)
        critical: Si la operación es crítica
        
    Returns:
        Resultado de la consulta
    """
    global divine_system
    
    if not divine_system.initialized:
        await divine_system.initialize()
        
    return await divine_system.execute_divine_query(
        sql=sql, 
        params=params, 
        priority=priority, 
        critical=critical
    )

async def execute_divine_transaction(queries: List[Dict[str, Any]], 
                                   priority: int = 8, 
                                   critical: bool = True) -> List[Any]:
    """
    Ejecutar transacción usando el sistema divino.
    
    Args:
        queries: Lista de consultas, cada una como un diccionario con keys 'query' y 'params'
        priority: Prioridad (1-10)
        critical: Si la operación es crítica
        
    Returns:
        Resultados de las consultas
    """
    global divine_system
    
    if not divine_system.initialized:
        await divine_system.initialize()
        
    return await divine_system.execute_divine_transaction(
        queries=queries,
        priority=priority,
        critical=critical
    )

async def get_divine_system_stats() -> Dict[str, Any]:
    """
    Obtener estadísticas completas del sistema divino.
    
    Returns:
        Estadísticas detalladas
    """
    global divine_system
    
    if not divine_system.initialized:
        return {"error": "Sistema divino no inicializado"}
        
    return await divine_system.get_system_stats()

# Inicializar el sistema automáticamente al importar
async def ensure_divine_system_ready():
    """Garantizar que el sistema divino esté inicializado y en ejecución."""
    global divine_system
    
    if not divine_system.initialized:
        await divine_system.initialize()
        
    if not divine_system.divine_queue.running:
        await divine_system.start()

# Crear tarea para inicialización en segundo plano
asyncio.create_task(ensure_divine_system_ready())