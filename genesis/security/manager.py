"""
Gestor de seguridad para el sistema Genesis.

Este módulo proporciona funcionalidades para garantizar la seguridad del sistema,
incluyendo encriptación, firma de transacciones, verificación de integridad, y
protección contra actividades maliciosas o no autorizadas.
"""

import base64
import hashlib
import hmac
import asyncio
from typing import Tuple, Dict, List, Any, Optional, Union
import time
import json
import logging
import os
import uuid
from datetime import datetime, timedelta

from genesis.core.base import Component
from genesis.utils.logger import setup_logging


class SecurityUtils:
    """Utilidades de seguridad."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Generar hash seguro para una contraseña.
        
        Args:
            password: Contraseña a hashear
            
        Returns:
            Hash de la contraseña
        """
        # Utilizar algoritmo de hasheo seguro (SHA-256)
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verificar si una contraseña coincide con su hash.
        
        Args:
            password: Contraseña a verificar
            password_hash: Hash almacenado
            
        Returns:
            True si la contraseña coincide, False en caso contrario
        """
        # Generar hash de la contraseña y comparar
        return SecurityUtils.hash_password(password) == password_hash
    
    @staticmethod
    def generate_signature(secret_key: str, message: str) -> str:
        """
        Generar firma HMAC para un mensaje.
        
        Args:
            secret_key: Clave secreta
            message: Mensaje a firmar
            
        Returns:
            Firma del mensaje
        """
        # Utilizar HMAC con SHA-256
        signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    @staticmethod
    def verify_signature(secret_key: str, message: str, signature: str) -> bool:
        """
        Verificar una firma HMAC.
        
        Args:
            secret_key: Clave secreta
            message: Mensaje firmado
            signature: Firma a verificar
            
        Returns:
            True si la firma es válida, False en caso contrario
        """
        # Generar firma del mensaje y comparar
        expected_signature = SecurityUtils.generate_signature(secret_key, message)
        return hmac.compare_digest(expected_signature, signature)
    
    @staticmethod
    def encrypt(secret_key: str, data: str) -> str:
        """
        Encriptar datos.
        
        Args:
            secret_key: Clave secreta
            data: Datos a encriptar
            
        Returns:
            Datos encriptados (base64)
        """
        # Generar key y nonce
        key = hashlib.sha256(secret_key.encode()).digest()
        nonce = os.urandom(16)
        
        # Importar solo si es necesario
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
        # Encriptar
        cipher = Cipher(
            algorithms.AES(key),
            modes.CTR(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data.encode()) + encryptor.finalize()
        
        # Combinar nonce y ciphertext, y convertir a base64
        return base64.b64encode(nonce + ciphertext).decode()
    
    @staticmethod
    def decrypt(secret_key: str, encrypted_data: str) -> str:
        """
        Desencriptar datos.
        
        Args:
            secret_key: Clave secreta
            encrypted_data: Datos encriptados (base64)
            
        Returns:
            Datos desencriptados
        """
        # Importar solo si es necesario
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
        # Decodificar base64
        data = base64.b64decode(encrypted_data)
        
        # Extraer nonce y ciphertext
        nonce = data[:16]
        ciphertext = data[16:]
        
        # Generar key
        key = hashlib.sha256(secret_key.encode()).digest()
        
        # Desencriptar
        cipher = Cipher(
            algorithms.AES(key),
            modes.CTR(nonce),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext.decode()
    
    @staticmethod
    def generate_api_key() -> Tuple[str, str]:
        """
        Generar par de claves API (pública y secreta).
        
        Returns:
            Tupla (api_key, api_secret)
        """
        # Generar API key (pública)
        api_key = str(uuid.uuid4()).replace("-", "")
        
        # Generar API secret (privada)
        api_secret = base64.b64encode(os.urandom(32)).decode()
        
        return api_key, api_secret


class SecurityManager(Component):
    """
    Gestor de seguridad del sistema.
    
    Este componente se encarga de gestionar aspectos de seguridad del sistema,
    como autenticación, encriptación, y detección de actividades sospechosas.
    """
    
    def __init__(self, name: str = "security_manager"):
        """
        Inicializar el gestor de seguridad.
        
        Args:
            name: Nombre del componente
        """
        super().__init__(name)
        self.logger = setup_logging(name)
        
        # Datos de usuarios y API keys
        self.users: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Tokens de acceso válidos
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Historial de intentos fallidos de autenticación
        self.failed_attempts: Dict[str, List[float]] = {}
        
        # Configuración
        self.token_expiry = 86400  # 24 horas en segundos
        self.max_failed_attempts = 5  # Máximo de intentos fallidos antes de bloqueo
        self.attempt_window = 3600  # Ventana de 1 hora para intentos fallidos
        self.blocked_ips: Dict[str, float] = {}  # IPs bloqueadas temporalmente
        self.block_duration = 3600  # Duración del bloqueo (1 hora)
        
        # Directorio de datos
        self.data_dir = "data/security"
        os.makedirs(self.data_dir, exist_ok=True)
    
    async def start(self) -> None:
        """Iniciar el gestor de seguridad."""
        await super().start()
        
        # Cargar datos
        self._load_users_data()
        self._load_api_keys_data()
        
        # Iniciar tareas periódicas
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Gestor de seguridad iniciado")
    
    async def stop(self) -> None:
        """Detener el gestor de seguridad."""
        # Cancelar tareas
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Guardar datos
        self._save_users_data()
        self._save_api_keys_data()
        
        await super().stop()
        self.logger.info("Gestor de seguridad detenido")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any], source: str) -> None:
        """
        Manejar eventos del bus de eventos.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            source: Componente de origen
        """
        # Verificar eventos de autenticación
        if event_type == "security.authenticate":
            await self._handle_authentication(data)
        
        # Verificar eventos de validación de token
        elif event_type == "security.validate_token":
            await self._handle_token_validation(data)
        
        # Verificar eventos de firma API
        elif event_type == "security.verify_api_signature":
            await self._handle_api_signature_verification(data)
    
    async def _handle_authentication(self, data: Dict[str, Any]) -> None:
        """
        Manejar solicitud de autenticación.
        
        Args:
            data: Datos de autenticación
        """
        username = data.get("username")
        password = data.get("password")
        ip_address = data.get("ip_address", "unknown")
        
        # Verificar si la IP está bloqueada
        if ip_address in self.blocked_ips:
            block_time = self.blocked_ips[ip_address]
            if time.time() - block_time < self.block_duration:
                # IP aún bloqueada
                await self.emit_event("security.authentication_response", {
                    "success": False,
                    "message": "IP bloqueada temporalmente por demasiados intentos fallidos",
                    "request_id": data.get("request_id")
                })
                return
            else:
                # Eliminar bloqueo expirado
                del self.blocked_ips[ip_address]
        
        # Verificar credenciales
        if username in self.users:
            user_data = self.users[username]
            password_hash = user_data.get("password_hash", "")
            
            if SecurityUtils.verify_password(password, password_hash):
                # Autenticación exitosa
                # Generar token de acceso
                token = str(uuid.uuid4())
                expiry = time.time() + self.token_expiry
                
                # Guardar token
                self.access_tokens[token] = {
                    "username": username,
                    "expiry": expiry,
                    "ip_address": ip_address
                }
                
                # Limpiar intentos fallidos
                if username in self.failed_attempts:
                    del self.failed_attempts[username]
                
                # Enviar respuesta
                await self.emit_event("security.authentication_response", {
                    "success": True,
                    "token": token,
                    "expiry": expiry,
                    "username": username,
                    "request_id": data.get("request_id")
                })
                
                # Registrar acceso exitoso
                self.logger.info(f"Autenticación exitosa: {username} desde {ip_address}")
                return
        
        # Autenticación fallida
        # Registrar intento fallido
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(time.time())
        
        # Limpiar intentos antiguos fuera de la ventana
        current_time = time.time()
        self.failed_attempts[username] = [
            t for t in self.failed_attempts[username]
            if current_time - t < self.attempt_window
        ]
        
        # Verificar si se excedió el límite de intentos
        if len(self.failed_attempts[username]) >= self.max_failed_attempts:
            # Bloquear IP
            self.blocked_ips[ip_address] = current_time
            
            # Enviar respuesta
            await self.emit_event("security.authentication_response", {
                "success": False,
                "message": "Demasiados intentos fallidos. IP bloqueada temporalmente.",
                "request_id": data.get("request_id")
            })
            
            # Emitir alerta de seguridad
            await self.emit_event("security.alert", {
                "type": "multiple_failed_logins",
                "username": username,
                "ip_address": ip_address,
                "attempts": len(self.failed_attempts[username]),
                "timestamp": current_time
            })
            
            self.logger.warning(f"IP bloqueada por múltiples intentos fallidos: {ip_address} para usuario {username}")
        else:
            # Enviar respuesta
            await self.emit_event("security.authentication_response", {
                "success": False,
                "message": "Credenciales inválidas",
                "request_id": data.get("request_id")
            })
            
            self.logger.info(f"Autenticación fallida: {username} desde {ip_address}")
    
    async def _handle_token_validation(self, data: Dict[str, Any]) -> None:
        """
        Manejar validación de token.
        
        Args:
            data: Datos de validación
        """
        token = data.get("token")
        ip_address = data.get("ip_address", "unknown")
        
        if token in self.access_tokens:
            token_data = self.access_tokens[token]
            
            # Verificar expiración
            if token_data["expiry"] > time.time():
                # Token válido
                await self.emit_event("security.token_validation_response", {
                    "valid": True,
                    "username": token_data["username"],
                    "request_id": data.get("request_id")
                })
                return
            else:
                # Token expirado
                del self.access_tokens[token]
        
        # Token inválido o expirado
        await self.emit_event("security.token_validation_response", {
            "valid": False,
            "message": "Token inválido o expirado",
            "request_id": data.get("request_id")
        })
    
    async def _handle_api_signature_verification(self, data: Dict[str, Any]) -> None:
        """
        Manejar verificación de firma API.
        
        Args:
            data: Datos de verificación
        """
        api_key = data.get("api_key")
        signature = data.get("signature")
        message = data.get("message")
        timestamp = data.get("timestamp", 0)
        ip_address = data.get("ip_address", "unknown")
        
        # Verificar que el API key exista
        if api_key not in self.api_keys:
            await self.emit_event("security.api_verification_response", {
                "valid": False,
                "message": "API key inválida",
                "request_id": data.get("request_id")
            })
            return
        
        api_data = self.api_keys[api_key]
        api_secret = api_data.get("api_secret", "")
        
        # Verificar que la API key esté activa
        if not api_data.get("active", False):
            await self.emit_event("security.api_verification_response", {
                "valid": False,
                "message": "API key inactiva",
                "request_id": data.get("request_id")
            })
            return
        
        # Verificar timestamp (máximo 5 minutos de diferencia)
        current_time = time.time()
        if abs(current_time - timestamp) > 300:
            await self.emit_event("security.api_verification_response", {
                "valid": False,
                "message": "Timestamp inválido",
                "request_id": data.get("request_id")
            })
            return
        
        # Verificar firma
        if SecurityUtils.verify_signature(api_secret, message, signature):
            # Firma válida
            await self.emit_event("security.api_verification_response", {
                "valid": True,
                "user_id": api_data.get("user_id"),
                "request_id": data.get("request_id")
            })
            
            # Registrar uso de API
            self.logger.info(f"API key utilizada correctamente: {api_key} desde {ip_address}")
        else:
            # Firma inválida
            await self.emit_event("security.api_verification_response", {
                "valid": False,
                "message": "Firma inválida",
                "request_id": data.get("request_id")
            })
            
            # Emitir alerta de seguridad
            await self.emit_event("security.alert", {
                "type": "invalid_api_signature",
                "api_key": api_key,
                "ip_address": ip_address,
                "timestamp": current_time
            })
            
            self.logger.warning(f"Firma API inválida: {api_key} desde {ip_address}")
    
    async def create_user(
        self,
        username: str,
        password: str,
        email: str,
        role: str = "user",
        active: bool = True
    ) -> bool:
        """
        Crear un nuevo usuario.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            email: Correo electrónico
            role: Rol del usuario (admin, user)
            active: Si el usuario está activo
            
        Returns:
            True si se creó correctamente, False en caso contrario
        """
        # Verificar si el usuario ya existe
        if username in self.users:
            self.logger.error(f"El usuario {username} ya existe")
            return False
        
        # Hashear contraseña
        password_hash = SecurityUtils.hash_password(password)
        
        # Crear usuario
        self.users[username] = {
            "username": username,
            "password_hash": password_hash,
            "email": email,
            "role": role,
            "active": active,
            "created_at": time.time(),
            "last_login": None
        }
        
        # Guardar datos
        self._save_users_data()
        
        self.logger.info(f"Usuario creado: {username}")
        return True
    
    async def update_user(
        self,
        username: str,
        password: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        active: Optional[bool] = None
    ) -> bool:
        """
        Actualizar un usuario existente.
        
        Args:
            username: Nombre de usuario
            password: Nueva contraseña (opcional)
            email: Nuevo correo electrónico (opcional)
            role: Nuevo rol (opcional)
            active: Nuevo estado (opcional)
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        # Verificar si el usuario existe
        if username not in self.users:
            self.logger.error(f"El usuario {username} no existe")
            return False
        
        # Obtener datos actuales
        user_data = self.users[username]
        
        # Actualizar datos
        if password is not None:
            user_data["password_hash"] = SecurityUtils.hash_password(password)
        
        if email is not None:
            user_data["email"] = email
        
        if role is not None:
            user_data["role"] = role
        
        if active is not None:
            user_data["active"] = active
        
        # Actualizar timestamp
        user_data["updated_at"] = time.time()
        
        # Guardar datos
        self._save_users_data()
        
        self.logger.info(f"Usuario actualizado: {username}")
        return True
    
    async def delete_user(self, username: str) -> bool:
        """
        Eliminar un usuario.
        
        Args:
            username: Nombre de usuario
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        # Verificar si el usuario existe
        if username not in self.users:
            self.logger.error(f"El usuario {username} no existe")
            return False
        
        # Eliminar usuario
        del self.users[username]
        
        # Eliminar tokens asociados
        for token, token_data in list(self.access_tokens.items()):
            if token_data["username"] == username:
                del self.access_tokens[token]
        
        # Eliminar API keys asociadas
        for key, key_data in list(self.api_keys.items()):
            if key_data["user_id"] == username:
                del self.api_keys[key]
        
        # Guardar datos
        self._save_users_data()
        self._save_api_keys_data()
        
        self.logger.info(f"Usuario eliminado: {username}")
        return True
    
    async def create_api_key(
        self,
        user_id: str,
        description: str,
        permissions: List[str],
        ip_whitelist: Optional[List[str]] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Crear un nuevo par de claves API.
        
        Args:
            user_id: ID del usuario
            description: Descripción de la clave
            permissions: Lista de permisos
            ip_whitelist: Lista de IPs permitidas (opcional)
            
        Returns:
            Tupla (api_key, api_secret) o None si hay error
        """
        # Verificar si el usuario existe
        if user_id not in self.users:
            self.logger.error(f"El usuario {user_id} no existe")
            return None
        
        # Generar par de claves
        api_key, api_secret = SecurityUtils.generate_api_key()
        
        # Almacenar API key
        self.api_keys[api_key] = {
            "api_key": api_key,
            "api_secret": api_secret,
            "user_id": user_id,
            "description": description,
            "permissions": permissions,
            "ip_whitelist": ip_whitelist or [],
            "active": True,
            "created_at": time.time(),
            "last_used": None
        }
        
        # Guardar datos
        self._save_api_keys_data()
        
        self.logger.info(f"API key creada para el usuario {user_id}")
        return api_key, api_secret
    
    async def update_api_key(
        self,
        api_key: str,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        ip_whitelist: Optional[List[str]] = None,
        active: Optional[bool] = None
    ) -> bool:
        """
        Actualizar una clave API existente.
        
        Args:
            api_key: Clave API a actualizar
            description: Nueva descripción (opcional)
            permissions: Nuevos permisos (opcional)
            ip_whitelist: Nueva lista de IPs permitidas (opcional)
            active: Nuevo estado (opcional)
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        # Verificar si la API key existe
        if api_key not in self.api_keys:
            self.logger.error(f"La API key {api_key} no existe")
            return False
        
        # Obtener datos actuales
        key_data = self.api_keys[api_key]
        
        # Actualizar datos
        if description is not None:
            key_data["description"] = description
        
        if permissions is not None:
            key_data["permissions"] = permissions
        
        if ip_whitelist is not None:
            key_data["ip_whitelist"] = ip_whitelist
        
        if active is not None:
            key_data["active"] = active
        
        # Actualizar timestamp
        key_data["updated_at"] = time.time()
        
        # Guardar datos
        self._save_api_keys_data()
        
        self.logger.info(f"API key actualizada: {api_key}")
        return True
    
    async def delete_api_key(self, api_key: str) -> bool:
        """
        Eliminar una clave API.
        
        Args:
            api_key: Clave API a eliminar
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        # Verificar si la API key existe
        if api_key not in self.api_keys:
            self.logger.error(f"La API key {api_key} no existe")
            return False
        
        # Eliminar API key
        del self.api_keys[api_key]
        
        # Guardar datos
        self._save_api_keys_data()
        
        self.logger.info(f"API key eliminada: {api_key}")
        return True
    
    async def logout(self, token: str) -> bool:
        """
        Cerrar sesión (invalidar token).
        
        Args:
            token: Token a invalidar
            
        Returns:
            True si se cerró sesión correctamente, False en caso contrario
        """
        # Verificar si el token existe
        if token in self.access_tokens:
            # Eliminar token
            del self.access_tokens[token]
            return True
        
        return False
    
    async def change_password(
        self,
        username: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """
        Cambiar contraseña de usuario.
        
        Args:
            username: Nombre de usuario
            current_password: Contraseña actual
            new_password: Nueva contraseña
            
        Returns:
            True si se cambió correctamente, False en caso contrario
        """
        # Verificar si el usuario existe
        if username not in self.users:
            self.logger.error(f"El usuario {username} no existe")
            return False
        
        # Verificar contraseña actual
        user_data = self.users[username]
        if not SecurityUtils.verify_password(current_password, user_data["password_hash"]):
            self.logger.error(f"Contraseña actual incorrecta para {username}")
            return False
        
        # Actualizar contraseña
        user_data["password_hash"] = SecurityUtils.hash_password(new_password)
        user_data["updated_at"] = time.time()
        
        # Guardar datos
        self._save_users_data()
        
        # Invalidar todos los tokens del usuario
        for token, token_data in list(self.access_tokens.items()):
            if token_data["username"] == username:
                del self.access_tokens[token]
        
        self.logger.info(f"Contraseña cambiada para el usuario {username}")
        return True
    
    async def _cleanup_loop(self) -> None:
        """Bucle de limpieza periódica."""
        while True:
            try:
                # Limpiar tokens expirados
                current_time = time.time()
                expired_tokens = [
                    token for token, data in self.access_tokens.items()
                    if data["expiry"] < current_time
                ]
                
                for token in expired_tokens:
                    del self.access_tokens[token]
                
                # Limpiar IPs bloqueadas
                expired_blocks = [
                    ip for ip, block_time in self.blocked_ips.items()
                    if current_time - block_time > self.block_duration
                ]
                
                for ip in expired_blocks:
                    del self.blocked_ips[ip]
                
                # Limpiar intentos fallidos antiguos
                for username in list(self.failed_attempts.keys()):
                    self.failed_attempts[username] = [
                        t for t in self.failed_attempts[username]
                        if current_time - t < self.attempt_window
                    ]
                    
                    if not self.failed_attempts[username]:
                        del self.failed_attempts[username]
                
            except asyncio.CancelledError:
                # El bucle fue cancelado
                break
            except Exception as e:
                self.logger.error(f"Error en bucle de limpieza: {e}")
            
            # Esperar para la próxima limpieza (cada hora)
            await asyncio.sleep(3600)
    
    def _load_users_data(self) -> None:
        """Cargar datos de usuarios desde archivo."""
        try:
            # Ruta del archivo
            file_path = f"{self.data_dir}/users.json"
            
            # Verificar si existe
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    self.users = json.load(f)
                
                self.logger.info(f"Cargados {len(self.users)} usuarios")
        
        except Exception as e:
            self.logger.error(f"Error al cargar datos de usuarios: {e}")
    
    def _save_users_data(self) -> None:
        """Guardar datos de usuarios a archivo."""
        try:
            # Ruta del archivo
            file_path = f"{self.data_dir}/users.json"
            
            # Guardar datos
            with open(file_path, "w") as f:
                json.dump(self.users, f, indent=2)
            
            self.logger.debug(f"Guardados {len(self.users)} usuarios")
        
        except Exception as e:
            self.logger.error(f"Error al guardar datos de usuarios: {e}")
    
    def _load_api_keys_data(self) -> None:
        """Cargar datos de API keys desde archivo."""
        try:
            # Ruta del archivo
            file_path = f"{self.data_dir}/api_keys.json"
            
            # Verificar si existe
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    self.api_keys = json.load(f)
                
                self.logger.info(f"Cargadas {len(self.api_keys)} API keys")
        
        except Exception as e:
            self.logger.error(f"Error al cargar datos de API keys: {e}")
    
    def _save_api_keys_data(self) -> None:
        """Guardar datos de API keys a archivo."""
        try:
            # Ruta del archivo
            file_path = f"{self.data_dir}/api_keys.json"
            
            # Guardar datos
            with open(file_path, "w") as f:
                json.dump(self.api_keys, f, indent=2)
            
            self.logger.debug(f"Guardadas {len(self.api_keys)} API keys")
        
        except Exception as e:
            self.logger.error(f"Error al guardar datos de API keys: {e}")


# Exportación para uso fácil
security_manager = SecurityManager()