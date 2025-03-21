"""
Gestor centralizado de logs para el sistema Genesis.

Este módulo proporciona un gestor centralizado para todos los logs del sistema,
incluyendo logs de trading, sistema, seguridad y auditoría, con soporte para
almacenamiento en base de datos y archivos.
"""

import os
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import traceback
from logging.handlers import RotatingFileHandler

# Configuración global
LOG_DIR = "data/logs"
LOG_DB_PATH = os.path.join(LOG_DIR, "genesis_logs.db")
LOG_LEVEL = logging.DEBUG

# Asegurar que existan los directorios
os.makedirs(LOG_DIR, exist_ok=True)


class JsonFormatter(logging.Formatter):
    """Formateador personalizado para logs en formato JSON."""
    
    def format(self, record):
        """
        Formatear un registro de log como JSON.
        
        Args:
            record: Registro de log
            
        Returns:
            String JSON con la información de log
        """
        # Información básica
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Añadir detalles de excepción si existe
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Añadir atributos extra
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", 
                           "filename", "funcName", "id", "levelname", "levelno", 
                           "lineno", "module", "msecs", "message", "msg", 
                           "name", "pathname", "process", "processName", 
                           "relativeCreated", "stack_info", "thread", "threadName"]:
                log_data[key] = value
        
        return json.dumps(log_data)


class DatabaseHandler(logging.Handler):
    """Handler personalizado para almacenar logs en SQLite."""
    
    def __init__(self, db_path: str):
        """
        Inicializar el handler de base de datos.
        
        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        super().__init__()
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Crear tablas si no existen
        self._init_db()
    
    def _init_db(self):
        """Inicializar la base de datos de logs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Crear tabla de logs
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT,
                    user TEXT,
                    correlation_id TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
                ''')
                
                # Crear índices
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_component ON logs(component)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_correlation_id ON logs(correlation_id)')
                
                conn.commit()
        except Exception as e:
            print(f"Error al inicializar la base de datos de logs: {e}")
    
    def emit(self, record):
        """
        Emitir un registro de log a la base de datos.
        
        Args:
            record: Registro de log
        """
        try:
            with self.lock:
                # Formatear el registro como JSON para procesar
                log_dict = json.loads(self.format(record))
                
                # Extraer datos
                timestamp = log_dict.get("timestamp", datetime.now().isoformat())
                level = log_dict.get("level", "INFO")
                component = log_dict.get("logger", "unknown")
                message = log_dict.get("message", "")
                source = log_dict.get("source", "system")
                user = log_dict.get("user", "system")
                correlation_id = log_dict.get("correlation_id")
                
                # Extraer metadata (eliminando campos ya procesados)
                metadata = {k: v for k, v in log_dict.items() 
                           if k not in ["timestamp", "level", "logger", "message", 
                                       "source", "user", "correlation_id"]}
                
                # Insertar en la base de datos
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    INSERT INTO logs 
                    (timestamp, level, component, message, source, user, correlation_id, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp,
                        level,
                        component,
                        message,
                        source,
                        user,
                        correlation_id,
                        json.dumps(metadata),
                        datetime.now().isoformat()
                    ))
                    conn.commit()
        except Exception as e:
            print(f"Error al escribir log en la base de datos: {e}")


class LogManager:
    """Gestor centralizado de logs para el sistema Genesis."""
    
    def __init__(self):
        """Inicializar el gestor de logs."""
        self.loggers = {}
        self.db_handler = None
        self.initialized = False
        
        # Lock para inicialización segura
        self.init_lock = threading.Lock()
    
    def initialize(self) -> None:
        """Inicializar el gestor de logs."""
        with self.init_lock:
            if self.initialized:
                return
            
            # Crear handler de base de datos
            self.db_handler = DatabaseHandler(LOG_DB_PATH)
            self.db_handler.setFormatter(JsonFormatter())
            self.db_handler.setLevel(LOG_LEVEL)
            
            # Configuración básica de logging
            logging.basicConfig(level=LOG_LEVEL)
            
            # Handler de archivo para logs generales
            file_handler = RotatingFileHandler(
                os.path.join(LOG_DIR, "genesis.log"),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            file_handler.setFormatter(JsonFormatter())
            file_handler.setLevel(LOG_LEVEL)
            
            # Añadir handlers al logger raíz
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.addHandler(self.db_handler)
            
            # Handler de archivo para logs de trading
            trading_file_handler = RotatingFileHandler(
                os.path.join(LOG_DIR, "trading.log"),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            trading_file_handler.setFormatter(JsonFormatter())
            trading_file_handler.setLevel(LOG_LEVEL)
            
            # Añadir handlers al logger de trading
            trading_logger = logging.getLogger("trading")
            trading_logger.addHandler(trading_file_handler)
            trading_logger.addHandler(self.db_handler)
            trading_logger.propagate = False  # No propagar al logger raíz
            
            # Handler de archivo para logs de seguridad
            security_file_handler = RotatingFileHandler(
                os.path.join(LOG_DIR, "security.log"),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            security_file_handler.setFormatter(JsonFormatter())
            security_file_handler.setLevel(LOG_LEVEL)
            
            # Añadir handlers al logger de seguridad
            security_logger = logging.getLogger("security")
            security_logger.addHandler(security_file_handler)
            security_logger.addHandler(self.db_handler)
            security_logger.propagate = False  # No propagar al logger raíz
            
            # Handler de archivo para logs de API
            api_file_handler = RotatingFileHandler(
                os.path.join(LOG_DIR, "api.log"),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            api_file_handler.setFormatter(JsonFormatter())
            api_file_handler.setLevel(LOG_LEVEL)
            
            # Añadir handlers al logger de API
            api_logger = logging.getLogger("api")
            api_logger.addHandler(api_file_handler)
            api_logger.addHandler(self.db_handler)
            api_logger.propagate = False  # No propagar al logger raíz
            
            # Handler de archivo para logs de transacciones
            transaction_file_handler = RotatingFileHandler(
                os.path.join(LOG_DIR, "transactions.log"),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            transaction_file_handler.setFormatter(JsonFormatter())
            transaction_file_handler.setLevel(LOG_LEVEL)
            
            # Añadir handlers al logger de transacciones
            transaction_logger = logging.getLogger("transactions")
            transaction_logger.addHandler(transaction_file_handler)
            transaction_logger.addHandler(self.db_handler)
            transaction_logger.propagate = False  # No propagar al logger raíz
            
            # Añadir el logger de auditoría
            audit_file_handler = RotatingFileHandler(
                os.path.join(LOG_DIR, "audit.log"),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            audit_file_handler.setFormatter(JsonFormatter())
            audit_file_handler.setLevel(LOG_LEVEL)
            
            # Añadir handlers al logger de auditoría
            audit_logger = logging.getLogger("audit")
            audit_logger.addHandler(audit_file_handler)
            audit_logger.addHandler(self.db_handler)
            audit_logger.propagate = False  # No propagar al logger raíz
            
            # Registrar loggers comunes
            self.loggers = {
                "root": root_logger,
                "trading": trading_logger,
                "security": security_logger,
                "api": api_logger,
                "transactions": transaction_logger,
                "audit": audit_logger
            }
            
            self.initialized = True
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtener un logger configurado.
        
        Args:
            name: Nombre del logger
            
        Returns:
            Logger configurado
        """
        # Inicializar si es necesario
        if not self.initialized:
            self.initialize()
        
        # Verificar si es un logger predefinido
        if name in self.loggers:
            return self.loggers[name]
        
        # Crear un nuevo logger
        logger = logging.getLogger(name)
        
        # Añadir el handler de base de datos si no lo tiene
        if self.db_handler not in [h for h in logger.handlers]:
            logger.addHandler(self.db_handler)
        
        return logger
    
    def log_event(
        self,
        level: str,
        message: str,
        component: str = "system",
        source: str = "system",
        user: str = "system",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar un evento en el log.
        
        Args:
            level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Mensaje a registrar
            component: Componente que origina el log
            source: Fuente del log
            user: Usuario que realiza la acción
            correlation_id: ID de correlación para agrupar logs relacionados
            metadata: Metadatos adicionales
        """
        # Inicializar si es necesario
        if not self.initialized:
            self.initialize()
        
        # Obtener nivel numérico
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Obtener logger para el componente
        logger = self.get_logger(component)
        
        # Crear extras
        extra = {
            "source": source,
            "user": user,
            "correlation_id": correlation_id
        }
        
        # Añadir metadatos si existen
        if metadata:
            extra.update(metadata)
        
        # Registrar
        logger.log(numeric_level, message, extra=extra)
    
    def query_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        level: Optional[str] = None,
        component: Optional[str] = None,
        user: Optional[str] = None,
        correlation_id: Optional[str] = None,
        search_text: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Consultar logs en la base de datos.
        
        Args:
            start_date: Fecha inicial (formato ISO)
            end_date: Fecha final (formato ISO)
            level: Nivel de log
            component: Componente
            user: Usuario
            correlation_id: ID de correlación
            search_text: Texto a buscar en el mensaje
            limit: Número máximo de resultados
            offset: Desplazamiento para paginación
            
        Returns:
            Lista de registros de log
        """
        # Inicializar si es necesario
        if not self.initialized:
            self.initialize()
        
        try:
            # Construir consulta SQL
            query = "SELECT * FROM logs WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            if level:
                query += " AND level = ?"
                params.append(level.upper())
            
            if component:
                query += " AND component = ?"
                params.append(component)
            
            if user:
                query += " AND user = ?"
                params.append(user)
            
            if correlation_id:
                query += " AND correlation_id = ?"
                params.append(correlation_id)
            
            if search_text:
                query += " AND message LIKE ?"
                params.append(f"%{search_text}%")
            
            # Ordenar y limitar
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.append(limit)
            params.append(offset)
            
            # Ejecutar consulta
            with sqlite3.connect(LOG_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convertir a diccionarios
                results = []
                for row in rows:
                    log_entry = dict(row)
                    
                    # Deserializar metadata
                    if "metadata" in log_entry and log_entry["metadata"]:
                        try:
                            log_entry["metadata"] = json.loads(log_entry["metadata"])
                        except:
                            log_entry["metadata"] = {}
                    
                    results.append(log_entry)
                
                return results
        
        except Exception as e:
            print(f"Error al consultar logs: {e}")
            return []
    
    def get_log_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas de logs.
        
        Args:
            start_date: Fecha inicial (formato ISO)
            end_date: Fecha final (formato ISO)
            
        Returns:
            Estadísticas de logs
        """
        # Inicializar si es necesario
        if not self.initialized:
            self.initialize()
        
        try:
            # Preparar consultas
            params = []
            date_condition = "1=1"
            
            if start_date:
                date_condition = "timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                date_condition += " AND timestamp <= ?" if start_date else "timestamp <= ?"
                params.append(end_date)
            
            # Ejecutar consultas
            with sqlite3.connect(LOG_DB_PATH) as conn:
                cursor = conn.cursor()
                
                # Total de logs
                cursor.execute(f"SELECT COUNT(*) FROM logs WHERE {date_condition}", params)
                total_logs = cursor.fetchone()[0]
                
                # Logs por nivel
                cursor.execute(f"""
                SELECT level, COUNT(*) 
                FROM logs 
                WHERE {date_condition}
                GROUP BY level
                """, params)
                level_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Logs por componente
                cursor.execute(f"""
                SELECT component, COUNT(*) 
                FROM logs 
                WHERE {date_condition}
                GROUP BY component
                """, params)
                component_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Logs por hora (últimas 24 horas)
                hour_params = params.copy()
                if not start_date and not end_date:
                    # Si no hay fechas, usar las últimas 24 horas
                    hour_condition = "timestamp >= datetime('now', '-1 day')"
                else:
                    hour_condition = date_condition
                
                cursor.execute(f"""
                SELECT strftime('%H', timestamp) as hour, COUNT(*) 
                FROM logs 
                WHERE {hour_condition}
                GROUP BY hour
                ORDER BY hour
                """, hour_params)
                hour_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_logs": total_logs,
                    "by_level": level_stats,
                    "by_component": component_stats,
                    "by_hour": hour_stats,
                    "period": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }
        
        except Exception as e:
            print(f"Error al obtener estadísticas de logs: {e}")
            return {}


# Singleton para uso global
log_manager = LogManager()


def get_logger(name: str) -> logging.Logger:
    """
    Obtener un logger configurado.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    return log_manager.get_logger(name)


def log_event(
    level: str,
    message: str,
    component: str = "system",
    source: str = "system",
    user: str = "system",
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Registrar un evento en el log.
    
    Args:
        level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Mensaje a registrar
        component: Componente que origina el log
        source: Fuente del log
        user: Usuario que realiza la acción
        correlation_id: ID de correlación para agrupar logs relacionados
        metadata: Metadatos adicionales
    """
    log_manager.log_event(level, message, component, source, user, correlation_id, metadata)


def query_logs(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    level: Optional[str] = None,
    component: Optional[str] = None,
    user: Optional[str] = None,
    correlation_id: Optional[str] = None,
    search_text: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Consultar logs en la base de datos.
    
    Args:
        start_date: Fecha inicial (formato ISO)
        end_date: Fecha final (formato ISO)
        level: Nivel de log
        component: Componente
        user: Usuario
        correlation_id: ID de correlación
        search_text: Texto a buscar en el mensaje
        limit: Número máximo de resultados
        offset: Desplazamiento para paginación
        
    Returns:
        Lista de registros de log
    """
    return log_manager.query_logs(
        start_date, end_date, level, component, user, 
        correlation_id, search_text, limit, offset
    )


def get_log_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Obtener estadísticas de logs.
    
    Args:
        start_date: Fecha inicial (formato ISO)
        end_date: Fecha final (formato ISO)
        
    Returns:
        Estadísticas de logs
    """
    return log_manager.get_log_stats(start_date, end_date)


# Inicializar el gestor de logs
log_manager.initialize()