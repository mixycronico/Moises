"""
Gestor de administradores e inversionistas para el Sistema Genesis.

Este módulo proporciona funcionalidades para gestionar usuarios
con roles de administrador e inversionista, permitiendo su creación,
modificación y autenticación.
"""

import logging
import os
import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import asyncio

from genesis.db.transcendental_database import transcendental_db

logger = logging.getLogger(__name__)

class UserRole:
    """Roles de usuario en el sistema."""
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"         # Super administrador
    FOUNDER_ADMIN = "founder_admin"     # Fundador con máximos privilegios (Moisés)
    INVESTOR = "investor"
    VIEWER = "viewer"
    ANALYST = "analyst"

class User:
    """Representación de un usuario en el sistema."""
    
    def __init__(
        self,
        username: str,
        email: str,
        password_hash: str,
        first_name: str = "",
        last_name: str = "",
        roles: Set[str] = None,
        user_id: str = None,
        created_at: float = None,
        last_login: float = None,
        preferences: Dict[str, Any] = None,
        active: bool = True
    ):
        """
        Inicializar usuario.
        
        Args:
            username: Nombre de usuario
            email: Correo electrónico
            password_hash: Hash de la contraseña
            first_name: Nombre
            last_name: Apellido
            roles: Conjunto de roles
            user_id: ID único del usuario
            created_at: Marca de tiempo de creación
            last_login: Marca de tiempo del último inicio de sesión
            preferences: Preferencias del usuario
            active: Estado de activación
        """
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.first_name = first_name
        self.last_name = last_name
        self.roles = roles or {UserRole.VIEWER}
        self.user_id = user_id or str(uuid.uuid4())
        self.created_at = created_at or datetime.now().timestamp()
        self.last_login = last_login
        self.preferences = preferences or {}
        self.active = active
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir usuario a diccionario.
        
        Returns:
            Diccionario con datos del usuario
        """
        return {
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "roles": list(self.roles),
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "preferences": self.preferences,
            "active": self.active
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """
        Crear usuario desde diccionario.
        
        Args:
            data: Diccionario con datos del usuario
            
        Returns:
            Instancia de usuario
        """
        roles = set(data.get("roles", [UserRole.VIEWER]))
        return cls(
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            roles=roles,
            user_id=data.get("user_id"),
            created_at=data.get("created_at"),
            last_login=data.get("last_login"),
            preferences=data.get("preferences", {}),
            active=data.get("active", True)
        )
        
    def has_role(self, role: str) -> bool:
        """
        Verificar si el usuario tiene un rol específico.
        
        Args:
            role: Rol a verificar
            
        Returns:
            True si tiene el rol, False en caso contrario
        """
        return role in self.roles
        
    def add_role(self, role: str) -> None:
        """
        Añadir rol al usuario.
        
        Args:
            role: Rol a añadir
        """
        self.roles.add(role)
        
    def remove_role(self, role: str) -> None:
        """
        Eliminar rol del usuario.
        
        Args:
            role: Rol a eliminar
        """
        if role in self.roles:
            self.roles.remove(role)
            
    def update_last_login(self) -> None:
        """Actualizar marca de tiempo de último inicio de sesión."""
        self.last_login = datetime.now().timestamp()
        
    def get_full_name(self) -> str:
        """
        Obtener nombre completo del usuario.
        
        Returns:
            Nombre completo
        """
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        return self.username

class AdminManager:
    """
    Gestor de administradores e inversionistas.
    
    Este componente gestiona usuarios con roles de administrador
    e inversionista en el sistema.
    """
    
    def __init__(
        self,
        name: str = "admin_manager"
    ):
        """
        Inicializar gestor de administradores.
        
        Args:
            name: Nombre del componente
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.users: Dict[str, User] = {}  # Por ID
        self.username_index: Dict[str, str] = {}  # username -> user_id
        self.email_index: Dict[str, str] = {}  # email -> user_id
        
    async def start(self) -> None:
        """Iniciar gestor de administradores."""
        # Cargar usuarios
        await self._load_users()
        
        self.logger.info(f"Gestor de administradores iniciado con {len(self.users)} usuarios")
        
    async def stop(self) -> None:
        """Detener gestor de administradores."""
        # Guardar usuarios
        await self._save_users()
        
        self.logger.info("Gestor de administradores detenido")
        
    async def _load_users(self) -> None:
        """Cargar usuarios desde la base de datos."""
        try:
            # Cargar desde la base de datos
            if transcendental_db:
                user_data = await transcendental_db.get_all(prefix="user:")
                
                for user_id, data in user_data.items():
                    user = User.from_dict(data)
                    self._add_user_to_memory(user)
                    
            # Si no hay usuarios, crear el administrador fundador por defecto
            if not self.users:
                self.logger.warning("No se encontraron usuarios. Creando administrador fundador por defecto...")
                user = await self.create_admin("Moisés Alvarenga", "mixycronico", "Jesus@2@7", "mixycronico@aol.com")
                if user:
                    user.add_role(UserRole.FOUNDER_ADMIN)
                    user.add_role(UserRole.SUPER_ADMIN)
                    await self._save_users()
                    self.logger.info("Configurado Moisés Alvarenga como FOUNDER_ADMIN")
                
        except Exception as e:
            self.logger.error(f"Error cargando usuarios: {e}")
            
    async def _save_users(self) -> None:
        """Guardar usuarios en la base de datos."""
        try:
            # Guardar en la base de datos
            if transcendental_db:
                for user_id, user in self.users.items():
                    await transcendental_db.store(f"user:{user_id}", user.to_dict())
                    
        except Exception as e:
            self.logger.error(f"Error guardando usuarios: {e}")
            
    def _add_user_to_memory(self, user: User) -> None:
        """
        Añadir usuario a los índices en memoria.
        
        Args:
            user: Usuario a añadir
        """
        self.users[user.user_id] = user
        self.username_index[user.username] = user.user_id
        self.email_index[user.email] = user.user_id
        
    def _remove_user_from_memory(self, user_id: str) -> None:
        """
        Eliminar usuario de los índices en memoria.
        
        Args:
            user_id: ID del usuario a eliminar
        """
        if user_id in self.users:
            user = self.users[user_id]
            del self.username_index[user.username]
            del self.email_index[user.email]
            del self.users[user_id]
            
    def _hash_password(self, password: str) -> str:
        """
        Generar hash de contraseña.
        
        Args:
            password: Contraseña en texto plano
            
        Returns:
            Hash de la contraseña
        """
        # Usar SHA-256 para hashing
        return hashlib.sha256(password.encode()).hexdigest()
        
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verificar contraseña.
        
        Args:
            password: Contraseña en texto plano
            password_hash: Hash de la contraseña
            
        Returns:
            True si la contraseña es correcta, False en caso contrario
        """
        return self._hash_password(password) == password_hash
        
    async def create_user(
        self,
        username: str,
        password: str,
        email: str,
        first_name: str = "",
        last_name: str = "",
        roles: Set[str] = None
    ) -> Optional[User]:
        """
        Crear un nuevo usuario.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            email: Correo electrónico
            first_name: Nombre
            last_name: Apellido
            roles: Conjunto de roles
            
        Returns:
            Usuario creado o None si hay error
        """
        # Verificar si el usuario ya existe
        if username in self.username_index or email in self.email_index:
            self.logger.error(f"Usuario o email ya existe: {username}, {email}")
            return None
            
        # Crear usuario
        password_hash = self._hash_password(password)
        roles = roles or {UserRole.VIEWER}
        
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            roles=roles
        )
        
        # Agregar a índices
        self._add_user_to_memory(user)
        
        # Guardar en base de datos
        await self._save_users()
        
        self.logger.info(f"Usuario creado: {username}, roles: {roles}")
        
        return user
        
    async def create_admin(
        self,
        full_name: str,
        username: str,
        password: str,
        email: str
    ) -> Optional[User]:
        """
        Crear un nuevo administrador.
        
        Args:
            full_name: Nombre completo
            username: Nombre de usuario
            password: Contraseña
            email: Correo electrónico
            
        Returns:
            Usuario creado o None si hay error
        """
        # Separar nombre y apellido
        name_parts = full_name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        return await self.create_user(
            username=username,
            password=password,
            email=email,
            first_name=first_name,
            last_name=last_name,
            roles={UserRole.ADMIN, UserRole.INVESTOR}
        )
        
    async def create_investor(
        self,
        full_name: str,
        username: str,
        password: str,
        email: str
    ) -> Optional[User]:
        """
        Crear un nuevo inversionista.
        
        Args:
            full_name: Nombre completo
            username: Nombre de usuario
            password: Contraseña
            email: Correo electrónico
            
        Returns:
            Usuario creado o None si hay error
        """
        # Separar nombre y apellido
        name_parts = full_name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        return await self.create_user(
            username=username,
            password=password,
            email=email,
            first_name=first_name,
            last_name=last_name,
            roles={UserRole.INVESTOR}
        )
        
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Autenticar usuario.
        
        Args:
            username: Nombre de usuario o email
            password: Contraseña
            
        Returns:
            Usuario autenticado o None si las credenciales son incorrectas
        """
        # Buscar por nombre de usuario
        user_id = self.username_index.get(username)
        
        # Si no se encuentra, buscar por email
        if not user_id:
            user_id = self.email_index.get(username)
            
        if not user_id or user_id not in self.users:
            return None
            
        user = self.users[user_id]
        
        # Verificar contraseña
        if not self._verify_password(password, user.password_hash):
            return None
            
        # Verificar roles especiales basados en el nombre o usuario
        full_name = user.get_full_name().lower()
        username = user.username.lower()
        
        # Moisés es siempre FOUNDER_ADMIN (máximo nivel)
        if "moises" in full_name or "alvarenga" in full_name or "mixycronico" in username:
            if not user.has_role(UserRole.FOUNDER_ADMIN):
                self.logger.info(f"Asignando rol de FOUNDER_ADMIN a {user.get_full_name()}")
                user.add_role(UserRole.FOUNDER_ADMIN)
                user.add_role(UserRole.SUPER_ADMIN)  # Los FOUNDER también son SUPER
                user.add_role(UserRole.ADMIN)  # Y también son ADMIN
                await self._save_users()
        
        # Jeremias Lazo y Stephany Sandoval son SUPER_ADMIN 
        elif "jeremias lazo" in full_name or "stephany sandoval" in full_name:
            if not user.has_role(UserRole.SUPER_ADMIN):
                self.logger.info(f"Asignando rol de SUPER_ADMIN a {user.get_full_name()}")
                user.add_role(UserRole.SUPER_ADMIN)
                user.add_role(UserRole.ADMIN)  # Asegurar que también tenga rol admin
                await self._save_users()
        
        # Actualizar último inicio de sesión
        user.update_last_login()
        await self._save_users()
        
        return user
        
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Obtener usuario por ID.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Usuario o None si no se encuentra
        """
        return self.users.get(user_id)
        
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Obtener usuario por nombre de usuario.
        
        Args:
            username: Nombre de usuario
            
        Returns:
            Usuario o None si no se encuentra
        """
        user_id = self.username_index.get(username)
        if user_id:
            return self.users.get(user_id)
        return None
        
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Obtener usuario por email.
        
        Args:
            email: Correo electrónico
            
        Returns:
            Usuario o None si no se encuentra
        """
        user_id = self.email_index.get(email)
        if user_id:
            return self.users.get(user_id)
        return None
        
    async def get_all_users(self) -> List[User]:
        """
        Obtener todos los usuarios.
        
        Returns:
            Lista de usuarios
        """
        return list(self.users.values())
        
    async def get_users_by_role(self, role: str) -> List[User]:
        """
        Obtener usuarios por rol.
        
        Args:
            role: Rol a filtrar
            
        Returns:
            Lista de usuarios con el rol especificado
        """
        return [user for user in self.users.values() if user.has_role(role)]
        
    async def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """
        Actualizar datos de usuario.
        
        Args:
            user_id: ID del usuario
            **kwargs: Datos a actualizar
            
        Returns:
            Usuario actualizado o None si no se encuentra
        """
        if user_id not in self.users:
            return None
            
        user = self.users[user_id]
        
        # Actualizar campos
        if "password" in kwargs:
            kwargs["password_hash"] = self._hash_password(kwargs.pop("password"))
        
        # Manejar cambios de nombre de usuario y email (índices)
        if "username" in kwargs and kwargs["username"] != user.username:
            del self.username_index[user.username]
            self.username_index[kwargs["username"]] = user_id
            
        if "email" in kwargs and kwargs["email"] != user.email:
            del self.email_index[user.email]
            self.email_index[kwargs["email"]] = user_id
            
        # Actualizar campos
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
                
        # Guardar cambios
        await self._save_users()
        
        return user
        
    async def delete_user(self, user_id: str) -> bool:
        """
        Eliminar usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        if user_id not in self.users:
            return False
            
        # Eliminar de índices en memoria
        self._remove_user_from_memory(user_id)
        
        # Eliminar de base de datos
        if transcendental_db:
            await transcendental_db.delete(f"user:{user_id}")
            
        return True
        
    async def add_role_to_user(self, user_id: str, role: str) -> bool:
        """
        Añadir rol a usuario.
        
        Args:
            user_id: ID del usuario
            role: Rol a añadir
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        if user_id not in self.users:
            return False
            
        user = self.users[user_id]
        user.add_role(role)
        
        # Guardar cambios
        await self._save_users()
        
        return True
        
    async def remove_role_from_user(self, user_id: str, role: str) -> bool:
        """
        Eliminar rol de usuario.
        
        Args:
            user_id: ID del usuario
            role: Rol a eliminar
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        if user_id not in self.users:
            return False
            
        user = self.users[user_id]
        user.remove_role(role)
        
        # Guardar cambios
        await self._save_users()
        
        return True
        
    async def promote_to_super_admin(self, promoter_id: str, target_user_id: str) -> Tuple[bool, str]:
        """
        Promover a un usuario a SUPER_ADMIN (solo puede hacerlo un FOUNDER_ADMIN).
        
        Args:
            promoter_id: ID del usuario que promueve (debe ser FOUNDER_ADMIN)
            target_user_id: ID del usuario a promover
            
        Returns:
            Tupla (éxito, mensaje)
        """
        # Verificar que el promotor existe y es FOUNDER_ADMIN
        if promoter_id not in self.users:
            return False, "Promotor no encontrado"
            
        promoter = self.users[promoter_id]
        if not promoter.has_role(UserRole.FOUNDER_ADMIN):
            return False, "No tienes permiso para promover a SUPER_ADMIN"
            
        # Verificar que el objetivo existe
        if target_user_id not in self.users:
            return False, "Usuario objetivo no encontrado"
            
        # Promover al usuario
        target_user = self.users[target_user_id]
        target_user.add_role(UserRole.SUPER_ADMIN)
        target_user.add_role(UserRole.ADMIN)  # Los SUPER_ADMIN también son ADMIN
        
        # Guardar cambios
        await self._save_users()
        
        self.logger.info(f"Usuario {target_user.get_full_name()} promovido a SUPER_ADMIN por {promoter.get_full_name()}")
        
        return True, f"{target_user.get_full_name()} ahora es SUPER_ADMIN"

# Instancia global
admin_manager: Optional[AdminManager] = None

async def initialize_admin_manager() -> Optional[AdminManager]:
    """
    Inicializar gestor de administradores.
    
    Returns:
        Gestor de administradores inicializado
    """
    global admin_manager
    
    try:
        admin_manager = AdminManager()
        await admin_manager.start()
        return admin_manager
    except Exception as e:
        logger.error(f"Error inicializando gestor de administradores: {e}")
        return None