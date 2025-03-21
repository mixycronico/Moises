"""
Componente de logging para la API REST del sistema Genesis.

Este módulo proporciona funcionalidad para registrar y analizar
las solicitudes y respuestas de la API REST.
"""

import os
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from flask import Flask, request, g

class APILogger:
    """Registrador de eventos para la API REST."""
    
    def __init__(self, app: Optional[Flask] = None, log_dir: str = "data/logs/api"):
        """
        Inicializar el logger de API.
        
        Args:
            app: Aplicación Flask (opcional)
            log_dir: Directorio para los archivos de log
        """
        self.logger = logging.getLogger('api')
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, "api.log")
        
        # Asegurar que el directorio de logs exista
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurar el logger
        handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        if app:
            self.init_app(app)
            
    def init_app(self, app: Flask) -> None:
        """
        Inicializar con una aplicación Flask.
        
        Args:
            app: Aplicación Flask
        """
        # Registrar hooks para solicitudes y respuestas
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.teardown_request(self._teardown_request)
        
    def _before_request(self) -> None:
        """Capturar información antes de procesar la solicitud."""
        # Generar ID para la solicitud
        g.request_id = str(uuid.uuid4())
        g.start_time = time.time()
        
        # Registrar la solicitud
        self.log_request(request.method, request.path, request.headers, request.get_data())
        
    def _after_request(self, response):
        """
        Capturar información después de procesar la solicitud.
        
        Args:
            response: Respuesta HTTP
            
        Returns:
            Respuesta HTTP
        """
        # Cálcular duración
        duration = time.time() - g.get('start_time', time.time())
        
        # Registrar la respuesta
        self.log_response(
            request.method, 
            request.path, 
            response.status_code,
            response.headers,
            response.get_data(),
            duration
        )
        
        return response
        
    def _teardown_request(self, exception) -> None:
        """
        Capturar errores durante el procesamiento.
        
        Args:
            exception: Excepción capturada (si la hay)
        """
        if exception:
            self.log_exception(request.method, request.path, exception)
            
    def log_request(
        self, 
        method: str, 
        path: str, 
        headers: Dict, 
        body: Any
    ) -> None:
        """
        Registrar una solicitud HTTP.
        
        Args:
            method: Método HTTP
            path: Ruta de la solicitud
            headers: Cabeceras HTTP
            body: Cuerpo de la solicitud
        """
        # Copiar headers sin información sensible
        safe_headers = dict(headers)
        if 'Authorization' in safe_headers:
            safe_headers['Authorization'] = '***REDACTED***'
            
        # Intentar decodificar el cuerpo si es JSON
        try:
            if isinstance(body, bytes):
                body_str = body.decode('utf-8')
                if body_str and body_str.strip():
                    body_data = json.loads(body_str)
                else:
                    body_data = None
            else:
                body_data = body
        except:
            body_data = "<binary data>"
            
        log_data = {
            'type': 'request',
            'data': {
                'request_id': g.get('request_id', str(uuid.uuid4())),
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'path': path,
                'headers': safe_headers,
                'body': body_data
            }
        }
        
        self._write_log(log_data)
        
    def log_response(
        self, 
        method: str, 
        path: str, 
        status_code: int,
        headers: Dict, 
        body: Any,
        duration: float
    ) -> None:
        """
        Registrar una respuesta HTTP.
        
        Args:
            method: Método HTTP
            path: Ruta de la solicitud
            status_code: Código de estado HTTP
            headers: Cabeceras HTTP
            body: Cuerpo de la respuesta
            duration: Duración en segundos
        """
        # Intentar decodificar el cuerpo si es JSON
        try:
            if isinstance(body, bytes):
                body_str = body.decode('utf-8')
                if body_str and body_str.strip():
                    body_data = json.loads(body_str)
                else:
                    body_data = None
            else:
                body_data = body
        except:
            body_data = "<binary data>"
            
        log_data = {
            'type': 'response',
            'data': {
                'request_id': g.get('request_id', str(uuid.uuid4())),
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'path': path,
                'status_code': status_code,
                'duration': duration,
                'headers': dict(headers),
                'body': body_data
            }
        }
        
        self._write_log(log_data)
        
    def log_exception(
        self, 
        method: str, 
        path: str, 
        exception: Exception
    ) -> None:
        """
        Registrar una excepción durante el procesamiento.
        
        Args:
            method: Método HTTP
            path: Ruta de la solicitud
            exception: Excepción capturada
        """
        log_data = {
            'type': 'exception',
            'data': {
                'request_id': g.get('request_id', str(uuid.uuid4())),
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'path': path,
                'exception_type': type(exception).__name__,
                'exception_message': str(exception)
            }
        }
        
        self._write_log(log_data)
        self.logger.error(f"API exception: {type(exception).__name__}: {str(exception)}")
        
    def _write_log(self, log_data: Dict) -> None:
        """
        Escribir un registro de log.
        
        Args:
            log_data: Datos para registrar
        """
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data) + '\n')
            
        # Registrar en el logger
        log_type = log_data.get('type', 'unknown')
        method = log_data.get('data', {}).get('method', 'UNKNOWN')
        path = log_data.get('data', {}).get('path', 'UNKNOWN')
        
        if log_type == 'request':
            self.logger.info(f"API request: {method} {path}")
        elif log_type == 'response':
            status = log_data.get('data', {}).get('status_code', 0)
            duration = log_data.get('data', {}).get('duration', 0)
            self.logger.info(f"API response: {method} {path} - {status} ({duration:.3f}s)")
            
    def get_recent_logs(
        self, 
        log_type: Optional[str] = None, 
        count: int = 50
    ) -> List[Dict]:
        """
        Obtener los logs más recientes.
        
        Args:
            log_type: Tipo de log (request, response, exception)
            count: Número máximo de logs
            
        Returns:
            Lista de registros de log
        """
        logs = []
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if not log_type or log_entry.get('type') == log_type:
                            logs.append(log_entry)
                    except:
                        continue
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            
        # Retornar los más recientes primero
        return sorted(
            logs, 
            key=lambda x: x.get('data', {}).get('timestamp', ''),
            reverse=True
        )[:count]


def initialize_logging(app: Flask) -> None:
    """
    Configurar logging global para la aplicación.
    
    Args:
        app: Aplicación Flask
    """
    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Asegurar que el directorio de logs exista
    os.makedirs('data/logs', exist_ok=True)
    
    # Handler para los logs de la aplicación
    app_handler = logging.FileHandler('data/logs/api.log')
    app_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(app_handler)
    
    # Handler para la consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    # Configurar nivel de log de bibliotecas externas
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logging.info(f"Logging inicializado para aplicación Flask")


def init_api_logger(app: Flask) -> APILogger:
    """
    Inicializar el logger de API para una aplicación Flask.
    
    Args:
        app: Aplicación Flask
        
    Returns:
        Instancia de APILogger configurada
    """
    api_logger = APILogger(app)
    app.api_logger = api_logger
    return api_logger