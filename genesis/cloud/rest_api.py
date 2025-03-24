#!/usr/bin/env python3
"""
REST API Ultra-Divina para componentes cloud del Sistema Genesis.

Este módulo implementa una API REST completa para interactuar con los
componentes cloud del Sistema Genesis: CircuitBreaker, CheckpointManager
y LoadBalancer, permitiendo la monitorización y control remotos, así como
la integración con sistemas externos.

La API incluye autenticación JWT, documentación OpenAPI, control de acceso
granular y capacidades de throttling para protegerse contra abusos.
"""

import os
import sys
import json
import logging
import time
import asyncio
import random
import uuid
import datetime
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from flask import Flask, request, jsonify, Blueprint, current_app
from flask_cors import CORS
from sqlalchemy.orm import scoped_session, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from functools import wraps

# Importaciones condicionales para Swagger/OpenAPI
try:
    from flasgger import Swagger, swag_from
    _HAS_SWAGGER = True
except ImportError:
    _HAS_SWAGGER = False
    def swag_from(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Importación segura de componentes cloud
try:
    from .circuit_breaker import (
        CloudCircuitBreaker, CloudCircuitBreakerFactory, CircuitState,
        circuit_breaker_factory, circuit_protected
    )
    from .distributed_checkpoint import (
        DistributedCheckpointManager, CheckpointStorageType, 
        CheckpointConsistencyLevel, CheckpointState, CheckpointMetadata,
        checkpoint_manager
    )
    from .load_balancer import (
        CloudLoadBalancer, CloudLoadBalancerManager, CloudNode,
        BalancerAlgorithm, ScalingPolicy, BalancerState,
        SessionAffinityMode, NodeHealthStatus,
        load_balancer_manager
    )
    _HAS_CLOUD_COMPONENTS = True
except ImportError:
    _HAS_CLOUD_COMPONENTS = False
    logging.warning("Componentes cloud no disponibles. API REST operará en modo limitado.")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("genesis.cloud.rest_api")


# Configuración de la API
DEFAULT_SECRET_KEY = os.environ.get("API_SECRET_KEY", str(uuid.uuid4()))
DEFAULT_TOKEN_EXPIRATION = 24 * 60 * 60  # 24 horas en segundos


class APIError(Exception):
    """Excepción personalizada para errores de API."""
    
    def __init__(self, message: str, status_code: int = 400, details: Any = None):
        """
        Inicializar error de API.
        
        Args:
            message: Mensaje de error
            status_code: Código de estado HTTP
            details: Detalles adicionales del error
        """
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class RateLimiter:
    """Limitador de tasa para prevenir abusos de la API."""
    
    def __init__(self, rate_limit: int = 60, time_window: int = 60):
        """
        Inicializar limitador.
        
        Args:
            rate_limit: Número máximo de solicitudes
            time_window: Ventana de tiempo en segundos
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.request_log: Dict[str, List[float]] = {}
    
    def check_limit(self, key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verificar si un cliente ha excedido su límite.
        
        Args:
            key: Identificador del cliente (IP, token, etc.)
            
        Returns:
            Tupla (permitido, info)
        """
        now = time.time()
        
        # Inicializar registro si no existe
        if key not in self.request_log:
            self.request_log[key] = []
        
        # Limpiar registros antiguos
        self.request_log[key] = [ts for ts in self.request_log[key] if now - ts <= self.time_window]
        
        # Verificar límite
        if len(self.request_log[key]) >= self.rate_limit:
            oldest = min(self.request_log[key]) if self.request_log[key] else now
            reset_time = oldest + self.time_window
            return False, {
                "limit": self.rate_limit,
                "remaining": 0,
                "reset": reset_time,
                "retry_after": max(1, int(reset_time - now))
            }
        
        # Registrar solicitud
        self.request_log[key].append(now)
        
        return True, {
            "limit": self.rate_limit,
            "remaining": self.rate_limit - len(self.request_log[key]),
            "reset": now + self.time_window
        }
    
    def cleanup(self) -> int:
        """
        Limpiar registros antiguos.
        
        Returns:
            Número de clientes eliminados
        """
        now = time.time()
        count_before = len(self.request_log)
        
        # Eliminar clientes sin solicitudes recientes
        to_delete = []
        for key, timestamps in self.request_log.items():
            self.request_log[key] = [ts for ts in timestamps if now - ts <= self.time_window]
            if not self.request_log[key]:
                to_delete.append(key)
        
        for key in to_delete:
            del self.request_log[key]
        
        return count_before - len(self.request_log)


class UserRole(Enum):
    """Roles de usuario para control de acceso."""
    VIEWER = 1       # Solo lectura
    OPERATOR = 2     # Puede realizar operaciones no destructivas
    ADMIN = 3        # Acceso completo
    SYSTEM = 4       # Para uso interno del sistema


class APIUser:
    """Clase para gestionar usuarios de la API."""
    
    def __init__(self, 
                 username: str, 
                 password_hash: str, 
                 role: UserRole,
                 api_key: Optional[str] = None,
                 enabled: bool = True):
        """
        Inicializar usuario.
        
        Args:
            username: Nombre de usuario
            password_hash: Hash de la contraseña
            role: Rol del usuario
            api_key: Clave API opcional
            enabled: Si el usuario está activo
        """
        self.username = username
        self.password_hash = password_hash
        self.role = role
        self.api_key = api_key or str(uuid.uuid4())
        self.enabled = enabled
        self.last_login = None
        self.created_at = time.time()
    
    def check_password(self, password: str) -> bool:
        """
        Verificar contraseña.
        
        Args:
            password: Contraseña a verificar
            
        Returns:
            True si es correcta
        """
        return check_password_hash(self.password_hash, password)
    
    @classmethod
    def create(cls, username: str, password: str, role: UserRole) -> 'APIUser':
        """
        Crear usuario nuevo.
        
        Args:
            username: Nombre de usuario
            password: Contraseña en texto plano
            role: Rol del usuario
            
        Returns:
            Usuario creado
        """
        password_hash = generate_password_hash(password)
        return cls(username, password_hash, role)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Datos del usuario
        """
        return {
            "username": self.username,
            "role": self.role.name,
            "api_key": "***" + self.api_key[-6:] if self.api_key else None,
            "enabled": self.enabled,
            "last_login": self.last_login,
            "created_at": datetime.datetime.fromtimestamp(self.created_at).isoformat()
        }


class UserManager:
    """Gestor de usuarios para la API."""
    
    def __init__(self):
        """Inicializar gestor de usuarios."""
        self.users: Dict[str, APIUser] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> username
    
    def add_user(self, user: APIUser) -> bool:
        """
        Añadir usuario.
        
        Args:
            user: Usuario a añadir
            
        Returns:
            True si se añadió correctamente
        """
        if user.username in self.users:
            return False
        
        self.users[user.username] = user
        if user.api_key:
            self.api_keys[user.api_key] = user.username
        
        return True
    
    def get_user(self, username: str) -> Optional[APIUser]:
        """
        Obtener usuario por nombre.
        
        Args:
            username: Nombre de usuario
            
        Returns:
            Usuario o None si no existe
        """
        return self.users.get(username)
    
    def get_user_by_api_key(self, api_key: str) -> Optional[APIUser]:
        """
        Obtener usuario por clave API.
        
        Args:
            api_key: Clave API
            
        Returns:
            Usuario o None si no existe
        """
        username = self.api_keys.get(api_key)
        if username:
            return self.users.get(username)
        return None
    
    def update_user(self, username: str, **kwargs) -> bool:
        """
        Actualizar usuario.
        
        Args:
            username: Nombre de usuario
            **kwargs: Atributos a actualizar
            
        Returns:
            True si se actualizó correctamente
        """
        user = self.users.get(username)
        if not user:
            return False
        
        # Actualizar atributos
        for key, value in kwargs.items():
            if key == "role" and isinstance(value, str):
                try:
                    value = UserRole[value]
                except KeyError:
                    continue
            
            if hasattr(user, key):
                setattr(user, key, value)
        
        # Actualizar registro de api_keys si cambió
        if "api_key" in kwargs:
            # Eliminar clave anterior
            old_keys = [k for k, u in self.api_keys.items() if u == username]
            for k in old_keys:
                self.api_keys.pop(k, None)
            
            # Añadir nueva clave
            if user.api_key:
                self.api_keys[user.api_key] = username
        
        return True
    
    def delete_user(self, username: str) -> bool:
        """
        Eliminar usuario.
        
        Args:
            username: Nombre de usuario
            
        Returns:
            True si se eliminó correctamente
        """
        user = self.users.pop(username, None)
        if not user:
            return False
        
        # Eliminar clave API
        if user.api_key and user.api_key in self.api_keys:
            self.api_keys.pop(user.api_key)
        
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[APIUser]:
        """
        Autenticar usuario con credenciales.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            
        Returns:
            Usuario autenticado o None
        """
        user = self.get_user(username)
        if user and user.enabled and user.check_password(password):
            user.last_login = time.time()
            return user
        return None
    
    def create_default_users(self) -> None:
        """Crear usuarios predeterminados si no existen."""
        if not self.users:
            # Usuario administrador
            admin = APIUser.create("admin", "admin123", UserRole.ADMIN)
            
            # Usuario operador
            operator = APIUser.create("operator", "operate123", UserRole.OPERATOR)
            
            # Usuario visor
            viewer = APIUser.create("viewer", "view123", UserRole.VIEWER)
            
            # Usuario sistema (para uso interno)
            system = APIUser.create("system", str(uuid.uuid4()), UserRole.SYSTEM)
            
            # Añadir usuarios
            self.add_user(admin)
            self.add_user(operator)
            self.add_user(viewer)
            self.add_user(system)
            
            logger.info("Usuarios predeterminados creados")


class CloudAPI:
    """API REST para componentes cloud del Sistema Genesis."""
    
    def __init__(self, 
                 app: Optional[Flask] = None,
                 url_prefix: str = "/api/cloud",
                 secret_key: Optional[str] = None,
                 enable_swagger: bool = True,
                 rate_limit: int = 60):
        """
        Inicializar API.
        
        Args:
            app: Aplicación Flask (opcional)
            url_prefix: Prefijo de URL para todas las rutas
            secret_key: Clave secreta para JWT
            enable_swagger: Si se debe habilitar Swagger
            rate_limit: Límite de solicitudes por minuto
        """
        self.url_prefix = url_prefix
        self.secret_key = secret_key or DEFAULT_SECRET_KEY
        self.token_expiration = DEFAULT_TOKEN_EXPIRATION
        self.enable_swagger = enable_swagger and _HAS_SWAGGER
        self.rate_limit = rate_limit
        
        # Gestor de usuarios
        self.user_manager = UserManager()
        
        # Limitador de tasa
        self.rate_limiter = RateLimiter(rate_limit=rate_limit)
        
        # Estado de inicialización
        self._initialized = False
        self._init_time = None
        
        # Registrar con aplicación si se proporciona
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app: Flask) -> None:
        """
        Inicializar con aplicación Flask.
        
        Args:
            app: Aplicación Flask
        """
        if self._initialized:
            return
        
        # Guardar tiempo de inicio
        self._init_time = time.time()
        
        # Guardar clave secreta en la aplicación
        app.config["JWT_SECRET_KEY"] = self.secret_key
        
        # Crear Blueprint
        bp = Blueprint("cloud_api", __name__, url_prefix=self.url_prefix)
        
        # Configurar CORS
        CORS(bp)
        
        # Configurar Swagger si está habilitado
        if self.enable_swagger:
            swagger_config = {
                "headers": [],
                "specs": [
                    {
                        "endpoint": "apispec",
                        "route": "/apispec.json",
                        "rule_filter": lambda rule: True,
                        "model_filter": lambda tag: True,
                    }
                ],
                "static_url_path": "/flasgger_static",
                "swagger_ui": True,
                "specs_route": "/docs/",
                "title": "Genesis Cloud API",
                "description": "API REST para componentes cloud del Sistema Genesis.",
                "version": "1.0.0",
                "termsOfService": "",
            }
            
            # Definiciones comunes para endpoints
            self.specs_dict = {
                "securityDefinitions": {
                    "Bearer": {
                        "type": "apiKey",
                        "name": "Authorization",
                        "in": "header",
                        "description": "JWT Authorization header using the Bearer scheme. Example: \"Authorization: Bearer {token}\""
                    },
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "name": "X-API-Key",
                        "in": "header",
                        "description": "API Key Authentication"
                    }
                },
                "security": [
                    {"Bearer": []},
                    {"ApiKeyAuth": []}
                ]
            }
            
            Swagger(app, config=swagger_config, template=self.specs_dict)
        
        # Crear usuarios predeterminados
        self.user_manager.create_default_users()
        
        # Obtener componentes cloud disponibles
        self.has_circuit_breaker = _HAS_CLOUD_COMPONENTS
        self.has_checkpoint_manager = _HAS_CLOUD_COMPONENTS
        self.has_load_balancer = _HAS_CLOUD_COMPONENTS
        
        # Inicializar componentes si están disponibles
        if _HAS_CLOUD_COMPONENTS:
            asyncio.run(self._init_cloud_components())
        
        # =====================================================================
        # Funciones auxiliares para protección de rutas
        # =====================================================================
        
        def get_token():
            """Obtener token de autorización."""
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                return auth_header.split(" ")[1]
            return None
        
        def get_api_key():
            """Obtener clave API."""
            return request.headers.get("X-API-Key")
        
        def get_client_id():
            """Obtener identificador de cliente para rate limiter."""
            # Usar token o API key si está disponible
            token = get_token()
            if token:
                try:
                    payload = jwt.decode(token, app.config["JWT_SECRET_KEY"], algorithms=["HS256"])
                    return f"token:{payload['sub']}"
                except:
                    pass
            
            api_key = get_api_key()
            if api_key:
                return f"api_key:{api_key}"
            
            # Usar IP como último recurso
            return f"ip:{request.remote_addr}"
        
        def rate_limit_exceeded():
            """Respuesta para límite excedido."""
            return jsonify({
                "error": "Rate limit exceeded",
                "status_code": 429,
                "message": "Too many requests. Please try again later."
            }), 429
        
        def check_rate_limit():
            """Verificar límite de tasa."""
            client_id = get_client_id()
            allowed, info = self.rate_limiter.check_limit(client_id)
            
            if not allowed:
                response = rate_limit_exceeded()
                response[0].headers["X-RateLimit-Limit"] = str(info["limit"])
                response[0].headers["X-RateLimit-Remaining"] = "0"
                response[0].headers["X-RateLimit-Reset"] = str(int(info["reset"]))
                response[0].headers["Retry-After"] = str(info["retry_after"])
                return response
            
            return None
        
        def handle_api_error(e):
            """Manejador de excepciones APIError."""
            response = jsonify({
                "error": str(e),
                "status_code": e.status_code,
                "message": e.message,
                "details": e.details
            })
            return response, e.status_code
        
        def token_required(f):
            """Decorador para requerir token válido."""
            @wraps(f)
            def decorated(*args, **kwargs):
                # Verificar límite de tasa
                rate_limit_response = check_rate_limit()
                if rate_limit_response:
                    return rate_limit_response
                
                token = get_token()
                if not token:
                    raise APIError("Token is missing", 401)
                
                try:
                    payload = jwt.decode(token, app.config["JWT_SECRET_KEY"], algorithms=["HS256"])
                    
                    # Verificar expiración
                    if "exp" in payload and payload["exp"] < time.time():
                        raise APIError("Token has expired", 401)
                    
                    # Obtener usuario
                    username = payload["sub"]
                    user = self.user_manager.get_user(username)
                    
                    if not user or not user.enabled:
                        raise APIError("Invalid or disabled user", 401)
                    
                    # Añadir información de usuario a la solicitud
                    kwargs["current_user"] = user
                    kwargs["user_role"] = user.role
                    
                except jwt.PyJWTError as e:
                    raise APIError(f"Invalid token: {str(e)}", 401)
                
                return f(*args, **kwargs)
            
            return decorated
        
        def api_key_required(f):
            """Decorador para requerir clave API válida."""
            @wraps(f)
            def decorated(*args, **kwargs):
                # Verificar límite de tasa
                rate_limit_response = check_rate_limit()
                if rate_limit_response:
                    return rate_limit_response
                
                api_key = get_api_key()
                if not api_key:
                    raise APIError("API key is missing", 401)
                
                user = self.user_manager.get_user_by_api_key(api_key)
                if not user or not user.enabled:
                    raise APIError("Invalid or disabled API key", 401)
                
                # Añadir información de usuario a la solicitud
                kwargs["current_user"] = user
                kwargs["user_role"] = user.role
                
                return f(*args, **kwargs)
            
            return decorated
        
        def auth_required(f):
            """Decorador que acepta token JWT o clave API."""
            @wraps(f)
            def decorated(*args, **kwargs):
                # Verificar límite de tasa
                rate_limit_response = check_rate_limit()
                if rate_limit_response:
                    return rate_limit_response
                
                # Intentar con token primero
                token = get_token()
                if token:
                    try:
                        payload = jwt.decode(token, app.config["JWT_SECRET_KEY"], algorithms=["HS256"])
                        
                        # Verificar expiración
                        if "exp" in payload and payload["exp"] < time.time():
                            # Intentar con API key si el token expiró
                            pass
                        else:
                            # Obtener usuario desde token
                            username = payload["sub"]
                            user = self.user_manager.get_user(username)
                            
                            if user and user.enabled:
                                # Añadir información de usuario a la solicitud
                                kwargs["current_user"] = user
                                kwargs["user_role"] = user.role
                                return f(*args, **kwargs)
                    except jwt.PyJWTError:
                        # Intentar con API key si el token es inválido
                        pass
                
                # Intentar con API key
                api_key = get_api_key()
                if api_key:
                    user = self.user_manager.get_user_by_api_key(api_key)
                    if user and user.enabled:
                        # Añadir información de usuario a la solicitud
                        kwargs["current_user"] = user
                        kwargs["user_role"] = user.role
                        return f(*args, **kwargs)
                
                # Si llegamos aquí, no hay autenticación válida
                raise APIError("Authentication required", 401)
            
            return decorated
        
        def role_required(min_role: UserRole):
            """
            Decorador para requerir un rol mínimo.
            
            Args:
                min_role: Rol mínimo requerido
            """
            def decorator(f):
                @wraps(f)
                def decorated(*args, **kwargs):
                    user_role = kwargs.get("user_role")
                    if not user_role or user_role.value < min_role.value:
                        raise APIError("Insufficient permissions", 403)
                    return f(*args, **kwargs)
                return decorated
            return decorator
        
        # =====================================================================
        # Definición de rutas de la API
        # =====================================================================
        
        @bp.errorhandler(APIError)
        def handle_error(e):
            return handle_api_error(e)
        
        @bp.route("/", methods=["GET"])
        def api_info():
            """Información general de la API."""
            info = {
                "name": "Genesis Cloud API",
                "version": "1.0.0",
                "status": "online",
                "uptime": time.time() - self._init_time,
                "components": {
                    "circuit_breaker": self.has_circuit_breaker,
                    "checkpoint_manager": self.has_checkpoint_manager,
                    "load_balancer": self.has_load_balancer
                }
            }
            return jsonify(info)
        
        @bp.route("/auth/login", methods=["POST"])
        def login():
            """
            Iniciar sesión para obtener token JWT.
            ---
            tags:
              - Authentication
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    username:
                      type: string
                    password:
                      type: string
            responses:
              200:
                description: Login successful
                schema:
                  type: object
                  properties:
                    token:
                      type: string
                    expires_at:
                      type: integer
                    user:
                      type: object
              401:
                description: Authentication failed
            """
            data = request.get_json()
            username = data.get("username")
            password = data.get("password")
            
            if not username or not password:
                raise APIError("Username and password are required", 400)
            
            user = self.user_manager.authenticate(username, password)
            if not user:
                raise APIError("Invalid username or password", 401)
            
            # Generar token
            expiration = int(time.time()) + self.token_expiration
            payload = {
                "sub": user.username,
                "role": user.role.name,
                "exp": expiration
            }
            
            token = jwt.encode(payload, app.config["JWT_SECRET_KEY"], algorithm="HS256")
            
            return jsonify({
                "token": token,
                "expires_at": expiration,
                "user": user.to_dict()
            })
        
        @bp.route("/auth/verify", methods=["GET"])
        @auth_required
        def verify_auth(current_user, user_role):
            """
            Verificar autenticación actual.
            ---
            tags:
              - Authentication
            security:
              - Bearer: []
              - ApiKeyAuth: []
            responses:
              200:
                description: Authentication valid
                schema:
                  type: object
                  properties:
                    user:
                      type: object
                    authenticated:
                      type: boolean
              401:
                description: Authentication invalid
            """
            return jsonify({
                "authenticated": True,
                "user": current_user.to_dict()
            })
        
        @bp.route("/auth/users", methods=["GET"])
        @auth_required
        @role_required(UserRole.ADMIN)
        def list_users(current_user, user_role):
            """
            Listar usuarios.
            ---
            tags:
              - Authentication
            security:
              - Bearer: []
              - ApiKeyAuth: []
            responses:
              200:
                description: List of users
                schema:
                  type: object
                  properties:
                    users:
                      type: array
                      items:
                        type: object
              403:
                description: Insufficient permissions
            """
            users = [user.to_dict() for user in self.user_manager.users.values()]
            return jsonify({"users": users})
        
        @bp.route("/auth/users", methods=["POST"])
        @auth_required
        @role_required(UserRole.ADMIN)
        def create_user(current_user, user_role):
            """
            Crear usuario.
            ---
            tags:
              - Authentication
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    username:
                      type: string
                    password:
                      type: string
                    role:
                      type: string
                      enum: [VIEWER, OPERATOR, ADMIN]
            responses:
              201:
                description: User created
                schema:
                  type: object
                  properties:
                    user:
                      type: object
              400:
                description: Invalid request
              403:
                description: Insufficient permissions
              409:
                description: User already exists
            """
            data = request.get_json()
            username = data.get("username")
            password = data.get("password")
            role_name = data.get("role", "VIEWER")
            
            if not username or not password:
                raise APIError("Username and password are required", 400)
            
            if username in self.user_manager.users:
                raise APIError("User already exists", 409)
            
            try:
                role = UserRole[role_name]
            except KeyError:
                raise APIError(f"Invalid role: {role_name}", 400)
            
            # Crear usuario
            user = APIUser.create(username, password, role)
            if not self.user_manager.add_user(user):
                raise APIError("Failed to create user", 500)
            
            return jsonify({"user": user.to_dict()}), 201
        
        @bp.route("/auth/users/<username>", methods=["PUT"])
        @auth_required
        @role_required(UserRole.ADMIN)
        def update_user(username, current_user, user_role):
            """
            Actualizar usuario.
            ---
            tags:
              - Authentication
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: username
                in: path
                required: true
                type: string
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    role:
                      type: string
                      enum: [VIEWER, OPERATOR, ADMIN]
                    enabled:
                      type: boolean
                    password:
                      type: string
            responses:
              200:
                description: User updated
                schema:
                  type: object
                  properties:
                    user:
                      type: object
              400:
                description: Invalid request
              403:
                description: Insufficient permissions
              404:
                description: User not found
            """
            data = request.get_json()
            
            # Solo permitir actualizar ciertos campos
            allowed_fields = {"role", "enabled", "password", "api_key"}
            update_data = {k: v for k, v in data.items() if k in allowed_fields}
            
            # Si hay contraseña, generar hash
            if "password" in update_data:
                update_data["password_hash"] = generate_password_hash(update_data.pop("password"))
            
            # Actualizar usuario
            user = self.user_manager.get_user(username)
            if not user:
                raise APIError("User not found", 404)
            
            if not self.user_manager.update_user(username, **update_data):
                raise APIError("Failed to update user", 500)
            
            # Obtener usuario actualizado
            user = self.user_manager.get_user(username)
            return jsonify({"user": user.to_dict()})
        
        @bp.route("/auth/users/<username>", methods=["DELETE"])
        @auth_required
        @role_required(UserRole.ADMIN)
        def delete_user(username, current_user, user_role):
            """
            Eliminar usuario.
            ---
            tags:
              - Authentication
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: username
                in: path
                required: true
                type: string
            responses:
              200:
                description: User deleted
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
              403:
                description: Insufficient permissions
              404:
                description: User not found
            """
            # No permitir eliminar usuario propio
            if username == current_user.username:
                raise APIError("Cannot delete own user", 400)
            
            user = self.user_manager.get_user(username)
            if not user:
                raise APIError("User not found", 404)
            
            if not self.user_manager.delete_user(username):
                raise APIError("Failed to delete user", 500)
            
            return jsonify({"success": True})
        
        @bp.route("/auth/rotate-api-key", methods=["POST"])
        @auth_required
        def rotate_api_key(current_user, user_role):
            """
            Generar nueva clave API para el usuario autenticado.
            ---
            tags:
              - Authentication
            security:
              - Bearer: []
              - ApiKeyAuth: []
            responses:
              200:
                description: API key rotated
                schema:
                  type: object
                  properties:
                    api_key:
                      type: string
                    user:
                      type: object
              401:
                description: Authentication failed
            """
            # Generar nueva clave API
            new_api_key = str(uuid.uuid4())
            
            # Actualizar usuario
            if not self.user_manager.update_user(current_user.username, api_key=new_api_key):
                raise APIError("Failed to update API key", 500)
            
            # Obtener usuario actualizado
            user = self.user_manager.get_user(current_user.username)
            
            return jsonify({
                "api_key": new_api_key,
                "user": user.to_dict()
            })
        
        # =====================================================================
        # Rutas para Circuit Breaker
        # =====================================================================
        
        @bp.route("/circuit-breakers", methods=["GET"])
        @auth_required
        def list_circuit_breakers(current_user, user_role):
            """
            Listar circuit breakers.
            ---
            tags:
              - Circuit Breaker
            security:
              - Bearer: []
              - ApiKeyAuth: []
            responses:
              200:
                description: List of circuit breakers
                schema:
                  type: object
                  properties:
                    circuit_breakers:
                      type: object
              404:
                description: Component not available
            """
            if not self.has_circuit_breaker:
                raise APIError("Circuit Breaker component not available", 404)
            
            cb_factory = circuit_breaker_factory
            circuit_breakers = {name: {
                "state": cb.get_state(),
                "metrics": cb.get_metrics()
            } for name, cb in cb_factory.get_all().items()}
            
            return jsonify({"circuit_breakers": circuit_breakers})
        
        @bp.route("/circuit-breakers/<name>", methods=["GET"])
        @auth_required
        def get_circuit_breaker(name, current_user, user_role):
            """
            Obtener circuit breaker.
            ---
            tags:
              - Circuit Breaker
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
            responses:
              200:
                description: Circuit breaker details
                schema:
                  type: object
                  properties:
                    name:
                      type: string
                    state:
                      type: string
                    metrics:
                      type: object
              404:
                description: Circuit breaker not found
            """
            if not self.has_circuit_breaker:
                raise APIError("Circuit Breaker component not available", 404)
            
            cb_factory = circuit_breaker_factory
            cb = cb_factory.get(name)
            
            if not cb:
                raise APIError(f"Circuit breaker '{name}' not found", 404)
            
            return jsonify({
                "name": name,
                "state": cb.get_state(),
                "metrics": cb.get_metrics()
            })
        
        @bp.route("/circuit-breakers", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def create_circuit_breaker(current_user, user_role):
            """
            Crear circuit breaker.
            ---
            tags:
              - Circuit Breaker
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    name:
                      type: string
                    failure_threshold:
                      type: integer
                    recovery_timeout:
                      type: number
                    half_open_capacity:
                      type: integer
                    quantum_failsafe:
                      type: boolean
            responses:
              201:
                description: Circuit breaker created
                schema:
                  type: object
                  properties:
                    name:
                      type: string
                    state:
                      type: string
              400:
                description: Invalid request
              404:
                description: Component not available
            """
            if not self.has_circuit_breaker:
                raise APIError("Circuit Breaker component not available", 404)
            
            data = request.get_json()
            name = data.get("name")
            
            if not name:
                raise APIError("Name is required", 400)
            
            # Parámetros opcionales
            failure_threshold = data.get("failure_threshold", 5)
            recovery_timeout = data.get("recovery_timeout", 0.000005)
            half_open_capacity = data.get("half_open_capacity", 2)
            quantum_failsafe = data.get("quantum_failsafe", True)
            
            # Crear circuit breaker
            cb_factory = circuit_breaker_factory
            cb = asyncio.run(cb_factory.create(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_capacity=half_open_capacity,
                quantum_failsafe=quantum_failsafe
            ))
            
            if not cb:
                raise APIError("Failed to create circuit breaker", 500)
            
            return jsonify({
                "name": name,
                "state": cb.get_state()
            }), 201
        
        @bp.route("/circuit-breakers/<name>/reset", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def reset_circuit_breaker(name, current_user, user_role):
            """
            Resetear circuit breaker.
            ---
            tags:
              - Circuit Breaker
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
            responses:
              200:
                description: Circuit breaker reset
                schema:
                  type: object
                  properties:
                    name:
                      type: string
                    state:
                      type: string
              404:
                description: Circuit breaker not found
            """
            if not self.has_circuit_breaker:
                raise APIError("Circuit Breaker component not available", 404)
            
            cb_factory = circuit_breaker_factory
            cb = cb_factory.get(name)
            
            if not cb:
                raise APIError(f"Circuit breaker '{name}' not found", 404)
            
            # Resetear circuit breaker
            asyncio.run(cb.reset())
            
            return jsonify({
                "name": name,
                "state": cb.get_state()
            })
        
        @bp.route("/circuit-breakers/<name>/force-open", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def force_open_circuit_breaker(name, current_user, user_role):
            """
            Forzar apertura de circuit breaker.
            ---
            tags:
              - Circuit Breaker
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
            responses:
              200:
                description: Circuit breaker forced open
                schema:
                  type: object
                  properties:
                    name:
                      type: string
                    state:
                      type: string
              404:
                description: Circuit breaker not found
            """
            if not self.has_circuit_breaker:
                raise APIError("Circuit Breaker component not available", 404)
            
            cb_factory = circuit_breaker_factory
            cb = cb_factory.get(name)
            
            if not cb:
                raise APIError(f"Circuit breaker '{name}' not found", 404)
            
            # Forzar apertura
            asyncio.run(cb.force_open())
            
            return jsonify({
                "name": name,
                "state": cb.get_state()
            })
        
        @bp.route("/circuit-breakers/<name>/force-closed", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def force_closed_circuit_breaker(name, current_user, user_role):
            """
            Forzar cierre de circuit breaker.
            ---
            tags:
              - Circuit Breaker
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
            responses:
              200:
                description: Circuit breaker forced closed
                schema:
                  type: object
                  properties:
                    name:
                      type: string
                    state:
                      type: string
              404:
                description: Circuit breaker not found
            """
            if not self.has_circuit_breaker:
                raise APIError("Circuit Breaker component not available", 404)
            
            cb_factory = circuit_breaker_factory
            cb = cb_factory.get(name)
            
            if not cb:
                raise APIError(f"Circuit breaker '{name}' not found", 404)
            
            # Forzar cierre
            asyncio.run(cb.force_closed())
            
            return jsonify({
                "name": name,
                "state": cb.get_state()
            })
        
        # =====================================================================
        # Rutas para Checkpoint Manager
        # =====================================================================
        
        @bp.route("/checkpoints", methods=["GET"])
        @auth_required
        def list_checkpoints(current_user, user_role):
            """
            Listar checkpoints.
            ---
            tags:
              - Checkpoint Manager
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: component_id
                in: query
                required: false
                type: string
            responses:
              200:
                description: List of checkpoints
                schema:
                  type: object
                  properties:
                    checkpoints:
                      type: array
                      items:
                        type: object
              404:
                description: Component not available
            """
            if not self.has_checkpoint_manager:
                raise APIError("Checkpoint Manager component not available", 404)
            
            component_id = request.args.get("component_id")
            
            checkpoints = asyncio.run(checkpoint_manager.list_checkpoints(component_id))
            
            # Convertir a diccionarios serializables
            checkpoint_dicts = []
            for cp in checkpoints:
                cp_dict = {
                    "checkpoint_id": cp.checkpoint_id,
                    "component_id": cp.component_id,
                    "timestamp": cp.timestamp,
                    "version": cp.version,
                    "consistency_level": cp.consistency_level.name,
                    "storage_type": cp.storage_type.name,
                    "state": cp.state.name,
                    "tags": cp.tags,
                    "dependencies": cp.dependencies,
                    "hash": cp.hash,
                    "size_bytes": cp.size_bytes,
                    "last_verified": cp.last_verified
                }
                checkpoint_dicts.append(cp_dict)
            
            return jsonify({"checkpoints": checkpoint_dicts})
        
        @bp.route("/checkpoints/<checkpoint_id>", methods=["GET"])
        @auth_required
        def get_checkpoint(checkpoint_id, current_user, user_role):
            """
            Obtener metadatos de checkpoint.
            ---
            tags:
              - Checkpoint Manager
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: checkpoint_id
                in: path
                required: true
                type: string
            responses:
              200:
                description: Checkpoint details
                schema:
                  type: object
                  properties:
                    checkpoint:
                      type: object
              404:
                description: Checkpoint not found
            """
            if not self.has_checkpoint_manager:
                raise APIError("Checkpoint Manager component not available", 404)
            
            # Cargar checkpoint sin datos
            _, metadata = asyncio.run(checkpoint_manager.load_checkpoint(checkpoint_id))
            
            if not metadata:
                raise APIError(f"Checkpoint '{checkpoint_id}' not found", 404)
            
            # Convertir a diccionario serializable
            checkpoint_dict = {
                "checkpoint_id": metadata.checkpoint_id,
                "component_id": metadata.component_id,
                "timestamp": metadata.timestamp,
                "version": metadata.version,
                "consistency_level": metadata.consistency_level.name,
                "storage_type": metadata.storage_type.name,
                "state": metadata.state.name,
                "tags": metadata.tags,
                "dependencies": metadata.dependencies,
                "hash": metadata.hash,
                "size_bytes": metadata.size_bytes,
                "last_verified": metadata.last_verified
            }
            
            return jsonify({"checkpoint": checkpoint_dict})
        
        @bp.route("/checkpoints/<checkpoint_id>", methods=["DELETE"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def delete_checkpoint(checkpoint_id, current_user, user_role):
            """
            Eliminar checkpoint.
            ---
            tags:
              - Checkpoint Manager
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: checkpoint_id
                in: path
                required: true
                type: string
            responses:
              200:
                description: Checkpoint deleted
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
              404:
                description: Checkpoint not found
            """
            if not self.has_checkpoint_manager:
                raise APIError("Checkpoint Manager component not available", 404)
            
            success = asyncio.run(checkpoint_manager.delete_checkpoint(checkpoint_id))
            
            if not success:
                raise APIError(f"Failed to delete checkpoint '{checkpoint_id}'", 500)
            
            return jsonify({"success": True})
        
        @bp.route("/checkpoints/cleanup", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def cleanup_checkpoints(current_user, user_role):
            """
            Limpiar checkpoints antiguos.
            ---
            tags:
              - Checkpoint Manager
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    component_id:
                      type: string
                    max_checkpoints:
                      type: integer
            responses:
              200:
                description: Checkpoints cleaned up
                schema:
                  type: object
                  properties:
                    deleted:
                      type: integer
              404:
                description: Component not available
            """
            if not self.has_checkpoint_manager:
                raise APIError("Checkpoint Manager component not available", 404)
            
            data = request.get_json()
            component_id = data.get("component_id")
            max_checkpoints = data.get("max_checkpoints", 5)
            
            deleted = asyncio.run(checkpoint_manager.cleanup_old_checkpoints(component_id, max_checkpoints))
            
            return jsonify({"deleted": deleted})
        
        # =====================================================================
        # Rutas para Load Balancer
        # =====================================================================
        
        @bp.route("/load-balancers", methods=["GET"])
        @auth_required
        def list_load_balancers(current_user, user_role):
            """
            Listar balanceadores de carga.
            ---
            tags:
              - Load Balancer
            security:
              - Bearer: []
              - ApiKeyAuth: []
            responses:
              200:
                description: List of load balancers
                schema:
                  type: object
                  properties:
                    load_balancers:
                      type: object
              404:
                description: Component not available
            """
            if not self.has_load_balancer:
                raise APIError("Load Balancer component not available", 404)
            
            # Obtener balanceadores
            balancers = load_balancer_manager.get_balancers_status()
            
            return jsonify({"load_balancers": balancers})
        
        @bp.route("/load-balancers/<name>", methods=["GET"])
        @auth_required
        def get_load_balancer(name, current_user, user_role):
            """
            Obtener balanceador de carga.
            ---
            tags:
              - Load Balancer
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
            responses:
              200:
                description: Load balancer details
                schema:
                  type: object
                  properties:
                    status:
                      type: object
                    nodes:
                      type: array
                      items:
                        type: object
              404:
                description: Load balancer not found
            """
            if not self.has_load_balancer:
                raise APIError("Load Balancer component not available", 404)
            
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(name)
            
            if not balancer:
                raise APIError(f"Load balancer '{name}' not found", 404)
            
            # Obtener estado y nodos
            status = balancer.get_status()
            nodes = balancer.get_nodes_status()
            
            return jsonify({
                "status": status,
                "nodes": nodes
            })
        
        @bp.route("/load-balancers", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def create_load_balancer(current_user, user_role):
            """
            Crear balanceador de carga.
            ---
            tags:
              - Load Balancer
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    name:
                      type: string
                    algorithm:
                      type: string
                      enum: [ROUND_ROBIN, LEAST_CONNECTIONS, WEIGHTED, RESPONSE_TIME, RESOURCE_BASED, PREDICTIVE, QUANTUM]
                    scaling_policy:
                      type: string
                      enum: [NONE, THRESHOLD, PREDICTIVE, SCHEDULE, ADAPTIVE]
                    session_affinity:
                      type: string
                      enum: [NONE, IP_BASED, COOKIE, TOKEN, HEADER, CONSISTENT_HASH]
            responses:
              201:
                description: Load balancer created
                schema:
                  type: object
                  properties:
                    name:
                      type: string
                    status:
                      type: object
              400:
                description: Invalid request
              404:
                description: Component not available
            """
            if not self.has_load_balancer:
                raise APIError("Load Balancer component not available", 404)
            
            data = request.get_json()
            name = data.get("name")
            
            if not name:
                raise APIError("Name is required", 400)
            
            # Parámetros opcionales
            algorithm_name = data.get("algorithm", "ROUND_ROBIN")
            scaling_policy_name = data.get("scaling_policy", "NONE")
            session_affinity_name = data.get("session_affinity", "NONE")
            
            # Convertir a enumeraciones
            try:
                algorithm = BalancerAlgorithm[algorithm_name]
                scaling_policy = ScalingPolicy[scaling_policy_name]
                session_affinity = SessionAffinityMode[session_affinity_name]
            except KeyError:
                raise APIError("Invalid algorithm, scaling policy, or session affinity", 400)
            
            # Crear balanceador
            balancer = asyncio.run(load_balancer_manager.create_balancer(
                name=name,
                algorithm=algorithm,
                scaling_policy=scaling_policy,
                session_affinity=session_affinity
            ))
            
            if not balancer:
                raise APIError("Failed to create load balancer", 500)
            
            # Obtener estado
            status = balancer.get_status()
            
            return jsonify({
                "name": name,
                "status": status
            }), 201
        
        @bp.route("/load-balancers/<name>/nodes", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def add_node(name, current_user, user_role):
            """
            Añadir nodo a balanceador.
            ---
            tags:
              - Load Balancer
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    node_id:
                      type: string
                    host:
                      type: string
                    port:
                      type: integer
                    weight:
                      type: number
                    max_connections:
                      type: integer
            responses:
              201:
                description: Node added
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
                    node:
                      type: object
              400:
                description: Invalid request
              404:
                description: Load balancer not found
            """
            if not self.has_load_balancer:
                raise APIError("Load Balancer component not available", 404)
            
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(name)
            
            if not balancer:
                raise APIError(f"Load balancer '{name}' not found", 404)
            
            data = request.get_json()
            node_id = data.get("node_id")
            host = data.get("host")
            port = data.get("port")
            
            if not node_id or not host or not port:
                raise APIError("Node ID, host, and port are required", 400)
            
            # Parámetros opcionales
            weight = data.get("weight", 1.0)
            max_connections = data.get("max_connections", 100)
            
            # Crear nodo
            node = CloudNode(
                node_id=node_id,
                host=host,
                port=port,
                weight=weight,
                max_connections=max_connections
            )
            
            # Añadir nodo
            success = asyncio.run(balancer.add_node(node))
            
            if not success:
                raise APIError("Failed to add node", 500)
            
            return jsonify({
                "success": True,
                "node": node.to_dict()
            }), 201
        
        @bp.route("/load-balancers/<name>/nodes/<node_id>", methods=["DELETE"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def remove_node(name, node_id, current_user, user_role):
            """
            Eliminar nodo de balanceador.
            ---
            tags:
              - Load Balancer
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
              - name: node_id
                in: path
                required: true
                type: string
            responses:
              200:
                description: Node removed
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
              404:
                description: Load balancer or node not found
            """
            if not self.has_load_balancer:
                raise APIError("Load Balancer component not available", 404)
            
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(name)
            
            if not balancer:
                raise APIError(f"Load balancer '{name}' not found", 404)
            
            # Eliminar nodo
            success = asyncio.run(balancer.remove_node(node_id))
            
            if not success:
                raise APIError("Failed to remove node", 500)
            
            return jsonify({"success": True})
        
        @bp.route("/load-balancers/<name>", methods=["DELETE"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def delete_load_balancer(name, current_user, user_role):
            """
            Eliminar balanceador de carga.
            ---
            tags:
              - Load Balancer
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
            responses:
              200:
                description: Load balancer deleted
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
              404:
                description: Load balancer not found
            """
            if not self.has_load_balancer:
                raise APIError("Load Balancer component not available", 404)
            
            # Eliminar balanceador
            success = asyncio.run(load_balancer_manager.delete_balancer(name))
            
            if not success:
                raise APIError("Failed to delete load balancer", 500)
            
            return jsonify({"success": True})
        
        @bp.route("/load-balancers/<name>/scale-up", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def scale_up(name, current_user, user_role):
            """
            Escalar balanceador hacia arriba.
            ---
            tags:
              - Load Balancer
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    count:
                      type: integer
            responses:
              200:
                description: Load balancer scaled up
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
                    status:
                      type: object
              404:
                description: Load balancer not found
            """
            if not self.has_load_balancer:
                raise APIError("Load Balancer component not available", 404)
            
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(name)
            
            if not balancer:
                raise APIError(f"Load balancer '{name}' not found", 404)
            
            data = request.get_json()
            count = data.get("count", 1)
            
            # Escalar
            success = asyncio.run(balancer._scale_up(count))
            
            if not success:
                raise APIError("Failed to scale up", 500)
            
            # Obtener estado actualizado
            status = balancer.get_status()
            
            return jsonify({
                "success": True,
                "status": status
            })
        
        @bp.route("/load-balancers/<name>/scale-down", methods=["POST"])
        @auth_required
        @role_required(UserRole.OPERATOR)
        def scale_down(name, current_user, user_role):
            """
            Escalar balanceador hacia abajo.
            ---
            tags:
              - Load Balancer
            security:
              - Bearer: []
              - ApiKeyAuth: []
            parameters:
              - name: name
                in: path
                required: true
                type: string
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    count:
                      type: integer
            responses:
              200:
                description: Load balancer scaled down
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
                    status:
                      type: object
              404:
                description: Load balancer not found
            """
            if not self.has_load_balancer:
                raise APIError("Load Balancer component not available", 404)
            
            # Obtener balanceador
            balancer = load_balancer_manager.get_balancer(name)
            
            if not balancer:
                raise APIError(f"Load balancer '{name}' not found", 404)
            
            data = request.get_json()
            count = data.get("count", 1)
            
            # Escalar
            success = asyncio.run(balancer._scale_down(count))
            
            if not success:
                raise APIError("Failed to scale down", 500)
            
            # Obtener estado actualizado
            status = balancer.get_status()
            
            return jsonify({
                "success": True,
                "status": status
            })
        
        @bp.route("/metrics", methods=["GET"])
        @auth_required
        def get_global_metrics(current_user, user_role):
            """
            Obtener métricas globales.
            ---
            tags:
              - Monitoring
            security:
              - Bearer: []
              - ApiKeyAuth: []
            responses:
              200:
                description: Global metrics
                schema:
                  type: object
                  properties:
                    metrics:
                      type: object
              404:
                description: Component not available
            """
            metrics = {}
            
            # Métricas de circuit breaker
            if self.has_circuit_breaker:
                cb_factory = circuit_breaker_factory
                circuit_breakers = {name: cb.get_metrics() for name, cb in cb_factory.get_all().items()}
                metrics["circuit_breakers"] = circuit_breakers
            
            # Métricas de load balancer
            if self.has_load_balancer:
                metrics["load_balancers"] = load_balancer_manager.get_global_metrics()
            
            # Métricas generales de API
            metrics["api"] = {
                "uptime": time.time() - self._init_time,
                "requests": len(self.rate_limiter.request_log),
                "clients": len(self.rate_limiter.request_log.keys())
            }
            
            return jsonify({"metrics": metrics})
        
        # =====================================================================
        # Manejador de errores global de Blueprint
        # =====================================================================
        
        @bp.errorhandler(Exception)
        def handle_exception(e):
            """Manejador de excepciones global."""
            if isinstance(e, APIError):
                return handle_api_error(e)
            
            # Excepciones no controladas
            logger.exception("Unhandled exception")
            response = jsonify({
                "error": "Internal server error",
                "status_code": 500,
                "message": str(e)
            })
            return response, 500
        
        # Registrar blueprint
        app.register_blueprint(bp)
        
        # Marcar inicialización completa
        self._initialized = True
        
        logger.info(f"Genesis Cloud API inicializada en {self.url_prefix}")
    
    async def _init_cloud_components(self) -> None:
        """Inicializar componentes cloud si no están inicializados."""
        # Inicializar load balancer manager
        if hasattr(load_balancer_manager, "initialize"):
            if not await load_balancer_manager.initialize():
                self.has_load_balancer = False
                logger.warning("Failed to initialize load balancer manager")


def create_cloud_api(app: Flask = None, url_prefix: str = "/api/cloud", enable_swagger: bool = True) -> CloudAPI:
    """
    Crear API REST para componentes cloud.
    
    Args:
        app: Aplicación Flask (opcional)
        url_prefix: Prefijo de URL para todas las rutas
        enable_swagger: Si se debe habilitar Swagger
        
    Returns:
        Instancia de CloudAPI
    """
    return CloudAPI(app=app, url_prefix=url_prefix, enable_swagger=enable_swagger)


# Para pruebas si se ejecuta este archivo directamente
if __name__ == "__main__":
    # Crear aplicación Flask
    app = Flask(__name__)
    
    # Crear API
    api = create_cloud_api(app, enable_swagger=True)
    
    # Configurar debug
    app.debug = True
    
    # Ejecutar servidor
    app.run(host="0.0.0.0", port=5000)