"""
Sistema de logs para la API REST.

Este módulo proporciona funcionalidades para registrar solicitudes y respuestas de API,
así como para el seguimiento de errores y auditoría.
"""

import time
import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import request, g, Flask
from werkzeug.exceptions import HTTPException

from genesis.utils.logger import setup_logging


# Configurar logger específico para API
api_logger = setup_logging('api_requests')


class APILogger:
    """
    Registrador de solicitudes y respuestas de API.
    
    Esta clase proporciona funcionalidades para registrar de manera estructurada
    las solicitudes y respuestas de la API, así como los tiempos de respuesta
    y errores que pudieran ocurrir durante el procesamiento.
    """
    
    def __init__(self, app: Optional[Flask] = None):
        """
        Inicializar el logger de API.
        
        Args:
            app: Aplicación Flask (opcional)
        """
        self.logger = api_logger
        
        # Registros en memoria para consulta rápida
        self.log_records: List[Dict[str, Any]] = []
        self.max_records = 1000  # Máximo de registros a mantener en memoria
        
        # Bloqueo para acceso concurrente a registros
        self.lock = threading.Lock()
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app: Flask) -> None:
        """
        Inicializar con una aplicación Flask.
        
        Args:
            app: Aplicación Flask
        """
        # Registrar before_request y after_request
        @app.before_request
        def before_request():
            """Acciones a realizar antes de procesar la solicitud."""
            # Generar un ID de correlación único para la solicitud
            g.correlation_id = str(uuid.uuid4())
            g.request_start_time = time.time()
            
            # Registrar detalles de la solicitud
            self.log_request()
        
        @app.after_request
        def after_request(response):
            """
            Acciones a realizar después de procesar la solicitud.
            
            Args:
                response: Respuesta de Flask
                
            Returns:
                Respuesta de Flask
            """
            # Registrar detalles de la respuesta
            self.log_response(response)
            return response
        
        @app.errorhandler(Exception)
        def handle_exception(e):
            """
            Manejar excepciones no capturadas.
            
            Args:
                e: Excepción
                
            Returns:
                Respuesta de error
            """
            # Verificar si es una excepción HTTP
            if isinstance(e, HTTPException):
                # Si es una excepción HTTP, seguir con el manejo normal
                return app.handle_http_exception(e)
            
            # Para otras excepciones, registrar el error
            self.log_exception(e)
            
            # Generar respuesta de error estándar
            response = {
                'success': False,
                'error': 'Internal Server Error',
                'message': str(e)
            }
            
            if app.debug:
                # En modo debug, incluir más detalles
                import traceback
                response['traceback'] = traceback.format_exc()
            
            return response, 500
        
        # Guardar referencia a la aplicación
        self.app = app
        self.logger.info(f"Logger de API inicializado para aplicación Flask")
    
    def log_request(self) -> None:
        """Registrar detalles de la solicitud actual."""
        try:
            # Extraer información relevante de la solicitud
            req_data = {
                'correlation_id': getattr(g, 'correlation_id', str(uuid.uuid4())),
                'timestamp': datetime.now().isoformat(),
                'method': request.method,
                'url': request.url,
                'path': request.path,
                'query_params': dict(request.args),
                'headers': {k: v for k, v in request.headers.items() 
                           if k.lower() not in ('authorization', 'cookie', 'x-api-key')},
                'remote_addr': request.remote_addr,
                'user_agent': request.user_agent.string,
                'content_type': request.content_type
            }
            
            # Añadir cuerpo de la solicitud si es JSON (evitando datos sensibles)
            if request.is_json:
                json_data = request.get_json(silent=True)
                if json_data:
                    # Eliminar campos sensibles
                    if isinstance(json_data, dict):
                        sanitized_data = json_data.copy()
                        for key in ['password', 'api_key', 'secret', 'token']:
                            if key in sanitized_data:
                                sanitized_data[key] = '***REDACTED***'
                        req_data['body'] = sanitized_data
            
            # Registrar la solicitud
            self.logger.info(
                f"API Request: {request.method} {request.path}",
                extra={'api_request': req_data}
            )
            
            # Almacenar en memoria
            with self.lock:
                self.log_records.append({
                    'type': 'request',
                    'data': req_data
                })
                
                # Limitar tamaño del registro en memoria
                if len(self.log_records) > self.max_records:
                    self.log_records = self.log_records[-self.max_records:]
        
        except Exception as e:
            self.logger.error(f"Error al registrar solicitud: {e}")
    
    def log_response(self, response) -> None:
        """
        Registrar detalles de la respuesta actual.
        
        Args:
            response: Respuesta de Flask
        """
        try:
            # Calcular tiempo de respuesta
            response_time = time.time() - getattr(g, 'request_start_time', time.time())
            
            # Extraer información relevante de la respuesta
            resp_data = {
                'correlation_id': getattr(g, 'correlation_id', str(uuid.uuid4())),
                'timestamp': datetime.now().isoformat(),
                'status_code': response.status_code,
                'response_time': response_time,
                'headers': {k: v for k, v in response.headers.items() 
                            if k.lower() not in ('set-cookie',)},
                'content_type': response.content_type
            }
            
            # Añadir cuerpo de la respuesta si es JSON
            if response.content_type and 'application/json' in response.content_type:
                try:
                    # Obtener una copia del cuerpo de la respuesta
                    response_data = json.loads(response.get_data(as_text=True))
                    resp_data['body'] = response_data
                except:
                    resp_data['body'] = '<invalid json>'
            
            # Registrar la respuesta
            log_level = 'info' if response.status_code < 400 else 'error'
            getattr(self.logger, log_level)(
                f"API Response: {response.status_code} ({response_time:.4f}s)",
                extra={'api_response': resp_data}
            )
            
            # Almacenar en memoria
            with self.lock:
                self.log_records.append({
                    'type': 'response',
                    'data': resp_data
                })
                
                # Limitar tamaño del registro en memoria
                if len(self.log_records) > self.max_records:
                    self.log_records = self.log_records[-self.max_records:]
        
        except Exception as e:
            self.logger.error(f"Error al registrar respuesta: {e}")
    
    def log_exception(self, exception: Exception) -> None:
        """
        Registrar detalles de una excepción.
        
        Args:
            exception: Excepción a registrar
        """
        try:
            import traceback
            
            # Extraer información relevante de la excepción
            ex_data = {
                'correlation_id': getattr(g, 'correlation_id', str(uuid.uuid4())),
                'timestamp': datetime.now().isoformat(),
                'exception_type': exception.__class__.__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc(),
                'request_method': request.method,
                'request_url': request.url,
                'request_path': request.path,
                'remote_addr': request.remote_addr
            }
            
            # Registrar la excepción
            self.logger.error(
                f"API Exception: {exception.__class__.__name__}: {str(exception)}",
                extra={'api_exception': ex_data}
            )
            
            # Almacenar en memoria
            with self.lock:
                self.log_records.append({
                    'type': 'exception',
                    'data': ex_data
                })
                
                # Limitar tamaño del registro en memoria
                if len(self.log_records) > self.max_records:
                    self.log_records = self.log_records[-self.max_records:]
        
        except Exception as e:
            self.logger.error(f"Error al registrar excepción: {e}")
    
    def get_recent_logs(
        self, 
        log_type: Optional[str] = None,
        count: int = 100,
        min_level: str = 'info'
    ) -> List[Dict[str, Any]]:
        """
        Obtener logs recientes almacenados en memoria.
        
        Args:
            log_type: Tipo de log (request, response, exception)
            count: Número máximo de logs a retornar
            min_level: Nivel mínimo a incluir
            
        Returns:
            Lista de registros de log
        """
        with self.lock:
            # Filtrar por tipo si se especifica
            filtered_logs = self.log_records
            if log_type:
                filtered_logs = [log for log in filtered_logs if log['type'] == log_type]
            
            # Retornar los más recientes (últimos)
            return filtered_logs[-count:]
    
    def clear_logs(self) -> None:
        """Limpiar registros en memoria."""
        with self.lock:
            self.log_records.clear()
            self.logger.info("Registros de API en memoria limpiados")


# Instancia global para uso fácil
api_logger_instance = APILogger()


def init_api_logger(app: Flask) -> APILogger:
    """
    Inicializar el logger de API para una aplicación Flask.
    
    Args:
        app: Aplicación Flask
        
    Returns:
        Instancia del logger de API
    """
    global api_logger_instance
    api_logger_instance.init_app(app)
    return api_logger_instance