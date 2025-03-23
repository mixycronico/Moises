"""
Módulo para interacción síncrona con la base de datos PostgreSQL.

Este módulo proporciona una interfaz síncrona para interactuar con la base de datos,
facilitando la integración con componentes que no pueden usar operaciones asíncronas.
"""
import os
import time
import logging
import psycopg2
from psycopg2 import extras
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configuración de logging
logger = logging.getLogger(__name__)

class SyncDatabase:
    """
    Interfaz síncrona para interacción con PostgreSQL.
    
    Esta clase proporciona métodos para ejecutar consultas SQL de forma síncrona,
    ideal para usar en contextos donde asyncio no está disponible o es problemático.
    """
    
    def __init__(self, db_url: Optional[str] = None, retries: int = 3, retry_delay: float = 0.5):
        """
        Inicializar la conexión a la base de datos.
        
        Args:
            db_url: URL de conexión a la base de datos (opcional, por defecto usa DATABASE_URL)
            retries: Número de reintentos para operaciones fallidas
            retry_delay: Tiempo de espera entre reintentos (segundos)
        """
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        if not self.db_url:
            logger.error("No se encontró URL de conexión a la base de datos")
            raise ValueError("DATABASE_URL no está definida")
            
        self.retries = retries
        self.retry_delay = retry_delay
        self._connection = None
        self._connect()
        
    def _connect(self) -> None:
        """Establecer conexión a la base de datos."""
        try:
            if self._connection is None or self._connection.closed:
                logger.debug(f"Conectando a PostgreSQL: {self.db_url[:20]}...'")
                self._connection = psycopg2.connect(self.db_url)
                logger.info("Conexión a PostgreSQL establecida correctamente")
        except Exception as e:
            logger.error(f"Error al conectar a PostgreSQL: {e}")
            raise
            
    def _ensure_connection(self) -> None:
        """Asegurar que la conexión está activa."""
        if self._connection is None or self._connection.closed:
            self._connect()
            
    def execute(self, sql: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None, 
                fetchall: bool = False, fetchone: bool = False) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Ejecutar una consulta SQL con reintentos automáticos.
        
        Args:
            sql: Consulta SQL a ejecutar
            params: Parámetros para la consulta (diccionario o tupla)
            fetchall: Si es True, retorna todos los resultados
            fetchone: Si es True, retorna solo el primer resultado
            
        Returns:
            Resultados de la consulta según las opciones de fetch, o None si no aplica
        """
        result = None
        attempts = 0
        last_error = None
        
        while attempts < self.retries:
            try:
                self._ensure_connection()
                
                # Usar cursor con diccionarios para retornar resultados más amigables
                with self._connection.cursor(cursor_factory=extras.RealDictCursor) as cursor:
                    cursor.execute(sql, params or ())
                    
                    if fetchall:
                        result = cursor.fetchall()
                    elif fetchone:
                        row = cursor.fetchone()
                        result = dict(row) if row else None
                    else:
                        self._connection.commit()
                        if cursor.rowcount >= 0:
                            result = cursor.rowcount
                            
                return result
                
            except Exception as e:
                last_error = e
                attempts += 1
                logger.warning(f"Error en ejecución SQL (intento {attempts}/{self.retries}): {e}")
                
                # Revertir cualquier cambio pendiente
                try:
                    self._connection.rollback()
                except Exception:
                    pass
                    
                # Cerrar y reintentar la conexión
                try:
                    self._connection.close()
                except Exception:
                    pass
                    
                self._connection = None
                
                if attempts < self.retries:
                    time.sleep(self.retry_delay)
        
        # Si llegamos aquí, fallaron todos los intentos
        logger.error(f"Fallo tras {self.retries} intentos de ejecutar SQL. Último error: {last_error}")
        raise last_error
        
    def fetch_all(self, sql: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None) -> List[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar todos los resultados.
        
        Args:
            sql: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Lista de diccionarios con los resultados
        """
        return self.execute(sql, params, fetchall=True) or []
        
    def fetch_one(self, sql: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None) -> Optional[Dict[str, Any]]:
        """
        Ejecutar consulta y retornar solo el primer resultado.
        
        Args:
            sql: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Diccionario con el primer resultado o None si no hay resultados
        """
        return self.execute(sql, params, fetchone=True)
        
    def fetch_val(self, sql: str, params: Optional[Union[Dict[str, Any], Tuple, List]] = None, 
                default: Any = None) -> Any:
        """
        Ejecutar consulta y retornar un único valor.
        
        Args:
            sql: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            default: Valor por defecto si no hay resultados
            
        Returns:
            Primer valor del primer resultado o default si no hay resultados
        """
        row = self.fetch_one(sql, params)
        if not row:
            return default
            
        # Retornar el primer valor
        return next(iter(row.values()), default)
        
    def execute_many(self, sql: str, params_list: List[Union[Dict[str, Any], Tuple, List]]) -> int:
        """
        Ejecutar la misma consulta con múltiples conjuntos de parámetros.
        
        Args:
            sql: Consulta SQL a ejecutar
            params_list: Lista de conjuntos de parámetros
            
        Returns:
            Número de filas afectadas
        """
        attempts = 0
        last_error = None
        
        while attempts < self.retries:
            try:
                self._ensure_connection()
                
                with self._connection.cursor() as cursor:
                    cursor.executemany(sql, params_list)
                    self._connection.commit()
                    return cursor.rowcount
                    
            except Exception as e:
                last_error = e
                attempts += 1
                logger.warning(f"Error en ejecución SQL múltiple (intento {attempts}/{self.retries}): {e}")
                
                # Revertir cualquier cambio pendiente
                try:
                    self._connection.rollback()
                except Exception:
                    pass
                    
                # Cerrar y reintentar la conexión
                try:
                    self._connection.close()
                except Exception:
                    pass
                    
                self._connection = None
                
                if attempts < self.retries:
                    time.sleep(self.retry_delay)
        
        # Si llegamos aquí, fallaron todos los intentos
        logger.error(f"Fallo tras {self.retries} intentos de ejecutar SQL múltiple. Último error: {last_error}")
        raise last_error
        
    def transaction(self, func: Callable) -> Any:
        """
        Ejecutar una función dentro de una transacción.
        
        Args:
            func: Función a ejecutar que recibe este objeto como parámetro
            
        Returns:
            Resultado de la función
        """
        self._ensure_connection()
        
        try:
            # Iniciar transacción
            self._connection.autocommit = False
            
            # Ejecutar función
            result = func(self)
            
            # Confirmar cambios
            self._connection.commit()
            return result
            
        except Exception as e:
            # Revertir cambios en caso de error
            logger.error(f"Error en transacción: {e}")
            self._connection.rollback()
            raise
            
        finally:
            # Restaurar autocommit
            self._connection.autocommit = True
            
    def close(self) -> None:
        """Cerrar la conexión a la base de datos."""
        if self._connection and not self._connection.closed:
            try:
                self._connection.close()
                logger.debug("Conexión a PostgreSQL cerrada correctamente")
            except Exception as e:
                logger.error(f"Error al cerrar conexión a PostgreSQL: {e}")
                
    def __del__(self) -> None:
        """Destructor para asegurar cierre de conexión."""
        self.close()
        
    def __enter__(self) -> 'SyncDatabase':
        """Para usar con with statement."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Para usar con with statement."""
        self.close()
        
    # Métodos de utilidad para operaciones comunes
    
    def insert(self, table: str, data: Dict[str, Any], returning: Optional[str] = None) -> Optional[Any]:
        """
        Insertar datos en una tabla.
        
        Args:
            table: Nombre de la tabla
            data: Diccionario con datos a insertar
            returning: Columna a retornar (opcional)
            
        Returns:
            Valor de la columna especificada en returning o None
        """
        columns = list(data.keys())
        placeholders = [f"%({col})s" for col in columns]
        
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        if returning:
            sql += f" RETURNING {returning}"
            return self.fetch_val(sql, data)
        else:
            self.execute(sql, data)
            return None
            
    def update(self, table: str, data: Dict[str, Any], condition: str, 
               condition_params: Optional[Dict[str, Any]] = None) -> int:
        """
        Actualizar datos en una tabla.
        
        Args:
            table: Nombre de la tabla
            data: Diccionario con datos a actualizar
            condition: Condición WHERE (con placeholders)
            condition_params: Parámetros para la condición
            
        Returns:
            Número de filas afectadas
        """
        set_items = [f"{col} = %({col})s" for col in data.keys()]
        
        sql = f"UPDATE {table} SET {', '.join(set_items)} WHERE {condition}"
        
        # Combinar parámetros de datos y condición
        params = {**data}
        if condition_params:
            params.update(condition_params)
            
        return self.execute(sql, params) or 0
        
    def delete(self, table: str, condition: str, params: Optional[Dict[str, Any]] = None) -> int:
        """
        Eliminar datos de una tabla.
        
        Args:
            table: Nombre de la tabla
            condition: Condición WHERE (con placeholders)
            params: Parámetros para la condición
            
        Returns:
            Número de filas afectadas
        """
        sql = f"DELETE FROM {table} WHERE {condition}"
        return self.execute(sql, params) or 0
        
    def table_exists(self, table: str) -> bool:
        """
        Verificar si una tabla existe.
        
        Args:
            table: Nombre de la tabla a verificar
            
        Returns:
            True si la tabla existe, False en caso contrario
        """
        sql = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
        """
        return self.fetch_val(sql, (table,), False) or False