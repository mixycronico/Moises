"""
Logger para la API REST.

Este módulo proporciona un sistema de logging específico para la API REST,
permitiendo registrar solicitudes, respuestas y errores.
"""

import os
import json
import logging
import time
from datetime import datetime
from flask import request, g
from typing import Dict, List, Any, Optional
from collections import deque


class APILogger:
    """Logger para la API REST."""
    
    def __init__(self, max_logs: int = 1000):
        """
        Inicializar el logger.
        
        Args:
            max_logs: Número máximo de logs a mantener en memoria
        """
        self.logger = logging.getLogger('api_logger')
        self.logger.setLevel(logging.INFO)
        
        # Configurar handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter para logs
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Agregar handler
        self.logger.addHandler(console_handler)
        
        # Registro en memoria
        self.request_logs = deque(maxlen=max_logs)
        self.response_logs = deque(maxlen=max_logs)
        self.exception_logs = deque(maxlen=max_logs)
        
        # Directorio de logs
        self.log_dir = os.path.join('data', 'logs', 'api')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Archivo de log
        log_file = os.path.join(self.log_dir, 'api.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Agregar file handler
        self.logger.addHandler(file_handler)
    
    def log_request(self, req=None):
        """
        Registrar una solicitud HTTP.
        
        Args:
            req: Objeto request de Flask (opcional)
        """
        if req is None:
            req = request
            
        # Obtener datos de la solicitud
        timestamp = datetime.utcnow().isoformat()
        path = req.path
        method = req.method
        args = dict(req.args)
        headers = dict(req.headers)
        
        # Filtrar headers sensibles
        sensitive_headers = ['Authorization', 'Cookie', 'X-Api-Key']
        for header in sensitive_headers:
            if header in headers:
                headers[header] = '[FILTERED]'
        
        # Datos de la solicitud
        request_data = {
            'timestamp': timestamp,
            'method': method,
            'path': path,
            'args': args,
            'headers': headers,
            'remote_addr': req.remote_addr,
            'endpoint': req.endpoint
        }
        
        # Intentar obtener el cuerpo de la solicitud
        try:
            if req.is_json:
                request_data['json'] = req.get_json()
        except Exception:
            request_data['json'] = None
        
        # Agregar a logs
        self.request_logs.append({
            'type': 'request',
            'data': request_data
        })
        
        # Guardar ID de solicitud para asociar con respuesta
        g.request_id = len(self.request_logs) - 1
        g.request_start_time = time.time()
        
        # Log en el logger
        self.logger.info(f"Request: {method} {path}")
        
        return request_data
    
    def log_response(self, response):
        """
        Registrar una respuesta HTTP.
        
        Args:
            response: Objeto response de Flask
            
        Returns:
            Objeto response sin modificar
        """
        # Obtener datos de la respuesta
        timestamp = datetime.utcnow().isoformat()
        status_code = response.status_code
        
        # Calcular tiempo de respuesta
        elapsed_time = time.time() - getattr(g, 'request_start_time', time.time())
        
        # Datos de la respuesta
        response_data = {
            'timestamp': timestamp,
            'status_code': status_code,
            'elapsed_time': elapsed_time,
            'content_type': response.content_type,
            'content_length': response.content_length,
            'request_id': getattr(g, 'request_id', None)
        }
        
        # Agregar a logs
        self.response_logs.append({
            'type': 'response',
            'data': response_data
        })
        
        # Log en el logger
        self.logger.info(f"Response: {status_code} in {elapsed_time:.4f}s")
        
        return response
    
    def log_exception(self, exception):
        """
        Registrar una excepción.
        
        Args:
            exception: Objeto de excepción
            
        Returns:
            Diccionario con datos de la excepción
        """
        # Obtener datos de la excepción
        timestamp = datetime.utcnow().isoformat()
        exception_type = type(exception).__name__
        exception_message = str(exception)
        
        # Datos de la excepción
        exception_data = {
            'timestamp': timestamp,
            'type': exception_type,
            'message': exception_message,
            'path': request.path if request else None,
            'method': request.method if request else None,
            'request_id': getattr(g, 'request_id', None)
        }
        
        # Agregar a logs
        self.exception_logs.append({
            'type': 'exception',
            'data': exception_data
        })
        
        # Log en el logger
        self.logger.error(f"Exception: {exception_type} - {exception_message}")
        
        return exception_data
    
    def get_recent_logs(self, log_type: Optional[str] = None, count: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener logs recientes.
        
        Args:
            log_type: Tipo de log (request, response, exception)
            count: Número máximo de logs a devolver
            
        Returns:
            Lista de logs
        """
        all_logs = []
        
        if log_type is None or log_type == 'request':
            all_logs.extend(list(self.request_logs))
            
        if log_type is None or log_type == 'response':
            all_logs.extend(list(self.response_logs))
            
        if log_type is None or log_type == 'exception':
            all_logs.extend(list(self.exception_logs))
        
        # Ordenar por timestamp
        all_logs.sort(key=lambda x: x['data'].get('timestamp', ''), reverse=True)
        
        return all_logs[:count]


def init_api_logger(app):
    """
    Inicializar el logger de la API.
    
    Args:
        app: Aplicación Flask
        
    Returns:
        Instancia de APILogger
    """
    # Crear instancia de APILogger
    api_logger = APILogger()
    
    # Registrar hooks para logging
    @app.before_request
    def before_request():
        api_logger.log_request()
    
    @app.after_request
    def after_request(response):
        return api_logger.log_response(response)
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        api_logger.log_exception(e)
        # Re-raise para que Flask maneje la excepción
        raise e
    
    # Guardar referencia en la aplicación
    app.api_logger = api_logger
    
    return api_logger