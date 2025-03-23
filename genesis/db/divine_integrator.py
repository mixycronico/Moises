"""
Integrador de la Cola Divina para el Sistema Genesis.

Este módulo proporciona funciones de integración para incorporar la cola divina
al Sistema Genesis, permitiendo una migración suave desde el sistema de tareas
de base de datos actual hacia la arquitectura híbrida Redis + RabbitMQ.

La integración se realiza de manera transparente, permitiendo que el sistema
funcione incluso cuando Redis o RabbitMQ no están disponibles.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Coroutine, Union

from .divine_task_queue import (
    DivineTaskQueue, 
    DivineConfig, 
    OperationMode,
    divine_task,
    critical_task,
    high_priority_task,
    low_priority_task,
    background_task,
    divine_transaction,
    initialize_divine_queue,
    ensure_queue_started
)

# Re-exportar decoradores para simplicidad
__all__ = [
    'divine_task', 'critical_task', 'high_priority_task', 
    'low_priority_task', 'background_task', 'divine_transaction',
    'initialize_divine_system', 'get_divine_stats', 'task_conversion',
    'DivineDatabaseOperations'
]

# Configuración de logging
logger = logging.getLogger("genesis.db.divine_integrator")

# Instancia global
_divine_system = None

async def initialize_divine_system(mode: str = OperationMode.DIVINO, auto_start: bool = True) -> DivineTaskQueue:
    """
    Inicializar el sistema divino con la configuración especificada.
    
    Args:
        mode: Modo de operación (normal, ultra, secure, adaptive, divine)
        auto_start: Iniciar automáticamente el sistema
        
    Returns:
        Instancia de DivineTaskQueue
    """
    global _divine_system
    
    # Obtener URLs de conexión desde variables de entorno o usar valores predeterminados
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    rabbitmq_url = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
    
    # Crear configuración
    config = DivineConfig(
        operation_mode=mode,
        redis_url=redis_url,
        rabbitmq_url=rabbitmq_url,
        redis_workers=4,
        rabbitmq_workers=2,
        enable_monitoring=True,
        auto_scaling=True
    )
    
    # Inicializar sistema
    _divine_system = initialize_divine_queue(config)
    
    # Inicializar y arrancar si es necesario
    if auto_start:
        await _divine_system.initialize()
        await _divine_system.start()
        logger.info(f"Sistema divino inicializado y arrancado en modo: {mode}")
    else:
        await _divine_system.initialize()
        logger.info(f"Sistema divino inicializado en modo: {mode} (sin arrancar)")
    
    return _divine_system

async def get_divine_stats() -> Dict[str, Any]:
    """
    Obtener estadísticas detalladas del sistema divino.
    
    Returns:
        Diccionario con estadísticas
    """
    global _divine_system
    
    if _divine_system is None:
        return {"error": "Sistema divino no inicializado"}
    
    try:
        return await _divine_system.get_stats_async()
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        return {"error": str(e)}

def task_conversion(priority: int = 5):
    """
    Decorador para convertir operaciones de base de datos existentes al sistema divino.
    
    Este decorador permite migrar funciones existentes al sistema divino
    sin cambiar su código interno, simplemente añadiendo este decorador.
    
    Args:
        priority: Prioridad de la tarea (1-10, 10 = máxima)
        
    Returns:
        Decorador configurado
    """
    def decorator(func):
        # Verificar si la función ya está decorada
        if hasattr(func, "__divine_converted__"):
            return func
            
        @divine_task(priority=priority)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
            
        async_wrapper.__divine_converted__ = True
        return async_wrapper
    
    return decorator

class DivineDatabaseOperations:
    """
    Clase de operaciones de base de datos divinas.
    
    Esta clase proporciona métodos de alto nivel para realizar
    operaciones comunes de base de datos utilizando el sistema divino.
    """
    
    @staticmethod
    @critical_task()
    async def save_critical_data(table: str, data: Dict[str, Any], db_session = None) -> Dict[str, Any]:
        """
        Guardar datos críticos con máxima prioridad y garantías.
        
        Args:
            table: Nombre de la tabla
            data: Datos a guardar
            db_session: Sesión de base de datos opcional
            
        Returns:
            Resultado de la operación
        """
        # Importamos aquí para evitar importaciones circulares
        from genesis.db.base import get_db_session
        
        session = db_session or await get_db_session()
        try:
            # Nota: Esta implementación es genérica, se debe adaptar según el ORM real
            query = f"INSERT INTO {table} ({', '.join(data.keys())}) VALUES ({', '.join(['%s'] * len(data))})"
            values = list(data.values())
            
            # Ejecutar consulta
            result = await session.execute(query, values)
            
            # Si no hay sesión externa, confirmar
            if not db_session:
                await session.commit()
                
            logger.info(f"Datos críticos guardados en {table}")
            return {"success": True, "result": result}
            
        except Exception as e:
            # Si no hay sesión externa, revertir
            if not db_session:
                await session.rollback()
                
            logger.error(f"Error al guardar datos críticos en {table}: {e}")
            raise
            
        finally:
            # Si no hay sesión externa, cerrar
            if not db_session:
                await session.close()
    
    @staticmethod
    @high_priority_task()
    async def update_data(table: str, data: Dict[str, Any], condition: Dict[str, Any], db_session = None) -> Dict[str, Any]:
        """
        Actualizar datos con alta prioridad.
        
        Args:
            table: Nombre de la tabla
            data: Datos a actualizar
            condition: Condición para la actualización
            db_session: Sesión de base de datos opcional
            
        Returns:
            Resultado de la operación
        """
        # Importamos aquí para evitar importaciones circulares
        from genesis.db.base import get_db_session
        
        session = db_session or await get_db_session()
        try:
            # Construir SET parte
            set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
            
            # Construir WHERE parte
            where_clause = " AND ".join([f"{k} = %s" for k in condition.keys()])
            
            # Construir query
            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
            
            # Valores para la consulta
            values = list(data.values()) + list(condition.values())
            
            # Ejecutar consulta
            result = await session.execute(query, values)
            
            # Si no hay sesión externa, confirmar
            if not db_session:
                await session.commit()
                
            logger.info(f"Datos actualizados en {table}")
            return {"success": True, "result": result}
            
        except Exception as e:
            # Si no hay sesión externa, revertir
            if not db_session:
                await session.rollback()
                
            logger.error(f"Error al actualizar datos en {table}: {e}")
            raise
            
        finally:
            # Si no hay sesión externa, cerrar
            if not db_session:
                await session.close()
                
    @staticmethod
    @divine_task(priority=5)
    async def read_data(table: str, condition: Dict[str, Any] = None, fields: List[str] = None, db_session = None) -> List[Dict[str, Any]]:
        """
        Leer datos con prioridad normal.
        
        Args:
            table: Nombre de la tabla
            condition: Condición para la lectura
            fields: Campos a seleccionar, None para todos
            db_session: Sesión de base de datos opcional
            
        Returns:
            Resultados de la consulta
        """
        # Importamos aquí para evitar importaciones circulares
        from genesis.db.base import get_db_session
        
        session = db_session or await get_db_session()
        try:
            # Construir SELECT parte
            select_clause = ", ".join(fields) if fields else "*"
            
            # Construir WHERE parte si hay condición
            where_clause = ""
            values = []
            if condition:
                where_clause = " WHERE " + " AND ".join([f"{k} = %s" for k in condition.keys()])
                values = list(condition.values())
            
            # Construir query
            query = f"SELECT {select_clause} FROM {table}{where_clause}"
            
            # Ejecutar consulta
            result = await session.fetch(query, *values)
            
            # Convertir a lista de diccionarios
            rows = [dict(row) for row in result]
            
            return rows
            
        except Exception as e:
            logger.error(f"Error al leer datos de {table}: {e}")
            raise
            
        finally:
            # Si no hay sesión externa, cerrar
            if not db_session:
                await session.close()
                
    @staticmethod
    @background_task()
    async def delete_data(table: str, condition: Dict[str, Any], db_session = None) -> Dict[str, Any]:
        """
        Eliminar datos en segundo plano.
        
        Args:
            table: Nombre de la tabla
            condition: Condición para la eliminación
            db_session: Sesión de base de datos opcional
            
        Returns:
            Resultado de la operación
        """
        # Importamos aquí para evitar importaciones circulares
        from genesis.db.base import get_db_session
        
        session = db_session or await get_db_session()
        try:
            # Construir WHERE parte
            where_clause = " AND ".join([f"{k} = %s" for k in condition.keys()])
            
            # Construir query
            query = f"DELETE FROM {table} WHERE {where_clause}"
            
            # Valores para la consulta
            values = list(condition.values())
            
            # Ejecutar consulta
            result = await session.execute(query, values)
            
            # Si no hay sesión externa, confirmar
            if not db_session:
                await session.commit()
                
            logger.info(f"Datos eliminados de {table}")
            return {"success": True, "result": result}
            
        except Exception as e:
            # Si no hay sesión externa, revertir
            if not db_session:
                await session.rollback()
                
            logger.error(f"Error al eliminar datos de {table}: {e}")
            raise
            
        finally:
            # Si no hay sesión externa, cerrar
            if not db_session:
                await session.close()
                
    @staticmethod
    @high_priority_task()
    async def execute_with_transaction(queries: List[Dict[str, Any]], db_session = None) -> Dict[str, Any]:
        """
        Ejecutar múltiples consultas en una transacción.
        
        Args:
            queries: Lista de consultas, cada una como un diccionario
                    con keys 'query' y 'values'
            db_session: Sesión de base de datos opcional
            
        Returns:
            Resultado de la operación
        """
        # Importamos aquí para evitar importaciones circulares
        from genesis.db.base import get_db_session
        
        session = db_session or await get_db_session()
        transaction = None
        try:
            # Iniciar transacción si no hay sesión externa
            if not db_session:
                transaction = await session.begin()
            
            results = []
            for query_data in queries:
                query = query_data["query"]
                values = query_data.get("values", [])
                
                # Ejecutar consulta
                if query.lower().startswith(("select", "with")):
                    result = await session.fetch(query, *values)
                    results.append([dict(row) for row in result])
                else:
                    result = await session.execute(query, *values)
                    results.append(result)
            
            # Confirmar transacción si la iniciamos nosotros
            if transaction and not db_session:
                await transaction.commit()
                
            return {"success": True, "results": results}
            
        except Exception as e:
            # Revertir transacción si la iniciamos nosotros
            if transaction and not db_session:
                await transaction.rollback()
                
            logger.error(f"Error en transacción: {e}")
            raise
            
        finally:
            # Cerrar sesión si la abrimos nosotros
            if not db_session:
                await session.close()
                
    @staticmethod
    async def with_transaction(func, *args, **kwargs):
        """
        Ejecutar una función dentro de una transacción divina.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos para la función
            **kwargs: Argumentos con nombre para la función
            
        Returns:
            Resultado de la función
        """
        async with divine_transaction():
            return await func(*args, **kwargs)