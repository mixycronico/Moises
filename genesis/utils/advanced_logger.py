"""
Sistema de logs avanzado para el sistema Genesis.

Este módulo proporciona un sistema de logs avanzado con soporte para
diferentes formatos, rotación de archivos, y almacenamiento en base de datos.
"""

import logging
import json
import sqlite3
import os
import uuid
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from genesis.config.settings import settings


# Nivel personalizado para auditoría
AUDIT = 25  # Entre INFO (20) y WARNING (30)
logging.addLevelName(AUDIT, "AUDIT")


# Configuración de base de datos para logs
DB_FILE = "data/logs/audit_log.db"


def init_log_db():
    """Inicializar la base de datos de logs."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audit_logs (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        level TEXT,
        message TEXT,
        component TEXT,
        user TEXT,
        metadata TEXT
    )''')
    conn.commit()
    conn.close()


# Asegurar que el directorio de logs existe
os.makedirs("logs", exist_ok=True)


class JsonFormatter(logging.Formatter):
    """Formateador personalizado que genera logs en formato JSON."""
    
    def format(self, record):
        """
        Formatear un registro de log como JSON.
        
        Args:
            record: Registro de log
            
        Returns:
            Cadena JSON
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
            "correlation_id": getattr(record, "correlation_id", str(uuid.uuid4())),
            "component": getattr(record, "component", "system"),
            "user": getattr(record, "user", "system"),
            "metadata": getattr(record, "metadata", {})
        }
        return json.dumps(log_entry)


class SQLiteHandler(logging.Handler):
    """Handler para almacenar logs en una base de datos SQLite."""
    
    def __init__(self):
        """Inicializar el handler de SQLite."""
        super().__init__()
        self.lock = threading.Lock()
        
        # Asegurar que la base de datos existe
        init_log_db()
    
    def emit(self, record):
        """
        Emitir un registro de log a la base de datos.
        
        Args:
            record: Registro de log
        """
        try:
            log_entry = json.loads(self.format(record))
            
            with self.lock:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute(
                    "INSERT INTO audit_logs (id, timestamp, level, message, component, user, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        log_entry["correlation_id"],
                        log_entry["timestamp"],
                        log_entry["level"],
                        log_entry["message"],
                        log_entry["component"],
                        log_entry["user"],
                        json.dumps(log_entry["metadata"])
                    )
                )
                conn.commit()
                conn.close()
        except Exception as e:
            # No podemos usar el log aquí para evitar recursión
            print(f"Error en SQLiteHandler: {e}")


def setup_advanced_logging(
    name: str,
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_db: bool = True,
    log_to_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    daily_rotation: bool = True
) -> logging.Logger:
    """
    Configurar un logger avanzado.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging (DEBUG, INFO, AUDIT, WARNING, ERROR, CRITICAL)
        log_to_file: Si se debe guardar en archivo
        log_to_db: Si se debe guardar en base de datos
        log_to_console: Si se debe mostrar en consola
        max_file_size: Tamaño máximo del archivo de log en bytes
        backup_count: Número de copias de respaldo
        daily_rotation: Si se debe hacer rotación diaria
        
    Returns:
        Logger configurado
    """
    # Convertir nivel de string a constante
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        if level.upper() == "AUDIT":
            numeric_level = AUDIT
        else:
            numeric_level = logging.INFO
    
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Crear formateador
    formatter = JsonFormatter()
    
    # Handler de archivo con rotación por tamaño
    if log_to_file:
        # Archivo principal
        log_file = f"logs/{name}.log"
        
        if daily_rotation:
            # Rotación diaria
            file_handler = TimedRotatingFileHandler(
                log_file, 
                when="midnight",
                interval=1,
                backupCount=backup_count
            )
        else:
            # Rotación por tamaño
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Handler de base de datos
    if log_to_db:
        db_handler = SQLiteHandler()
        db_handler.setFormatter(formatter)
        logger.addHandler(db_handler)
    
    # Handler de consola
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_action(
    logger: logging.Logger,
    level: str,
    message: str,
    component: str = "system",
    user: str = "system",
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Registrar una acción en el log.
    
    Args:
        logger: Logger a utilizar
        level: Nivel de log (DEBUG, INFO, AUDIT, WARNING, ERROR, CRITICAL)
        message: Mensaje a registrar
        component: Componente que origina el log
        user: Usuario que realiza la acción
        correlation_id: ID de correlación para agrupar logs relacionados
        metadata: Metadatos adicionales
    """
    # Crear o usar correlation_id
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Preparar datos extra
    extra = {
        "correlation_id": correlation_id,
        "component": component,
        "user": user,
        "metadata": metadata or {}
    }
    
    # Seleccionar método de log según el nivel
    if level.upper() == "AUDIT":
        logger.log(AUDIT, message, extra=extra)
    else:
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message, extra=extra)


def get_audit_logs(
    start_date: str,
    end_date: str,
    level: Optional[str] = None,
    component: Optional[str] = None,
    user: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Obtener logs de auditoría de la base de datos.
    
    Args:
        start_date: Fecha inicial (ISO format)
        end_date: Fecha final (ISO format)
        level: Filtrar por nivel (opcional)
        component: Filtrar por componente (opcional)
        user: Filtrar por usuario (opcional)
        limit: Límite de resultados
        
    Returns:
        Lista de registros de log
    """
    # Construir consulta SQL
    query = "SELECT * FROM audit_logs WHERE timestamp BETWEEN ? AND ?"
    params = [start_date, end_date]
    
    # Añadir filtros
    if level:
        query += " AND level = ?"
        params.append(level)
    
    if component:
        query += " AND component = ?"
        params.append(component)
    
    if user:
        query += " AND user = ?"
        params.append(user)
    
    # Ordenar y limitar
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(query, params)
        rows = c.fetchall()
        conn.close()
        
        # Convertir a dicts
        result = []
        for row in rows:
            metadata = {}
            try:
                metadata = json.loads(row[6])  # Índice de metadata
            except Exception:
                pass
            
            result.append({
                "id": row[0],
                "timestamp": row[1],
                "level": row[2],
                "message": row[3],
                "component": row[4],
                "user": row[5],
                "metadata": metadata
            })
        
        return result
    except Exception as e:
        # Registrar el error y devolver lista vacía
        print(f"Error al obtener logs: {e}")
        return []


# Ejemplo de uso:
if __name__ == "__main__":
    # Configurar logger
    test_logger = setup_advanced_logging("test_logger")
    
    # Log simple
    test_logger.info("Este es un mensaje de prueba")
    
    # Log con datos extra
    log_action(
        test_logger,
        "AUDIT",
        "Usuario realizó login",
        component="auth",
        user="test_user",
        metadata={"ip": "127.0.0.1", "user_agent": "Test Browser"}
    )
    
    # Obtener logs
    logs = get_audit_logs("2023-01-01T00:00:00", "2025-12-31T23:59:59")
    for log in logs:
        print(log)