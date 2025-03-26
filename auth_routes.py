"""
Rutas de autenticación para el Sistema Genesis.

Este módulo implementa los endpoints de API para la autenticación
de usuarios en el Sistema Genesis.
"""

from flask import jsonify, request, session

# Usuarios simulados para demo
DEMO_USERS = {
    "investor": {
        "id": 1,
        "username": "investor",
        "password": "investor_password",
        "email": "investor@genesis.ai",
        "role": "investor",
        "name": "Usuario Inversor"
    },
    "admin": {
        "id": 2,
        "username": "admin",
        "password": "admin_password",
        "email": "admin@genesis.ai",
        "role": "admin",
        "name": "Usuario Administrador"
    },
    "super_admin": {
        "id": 3,
        "username": "super_admin",
        "password": "super_admin_password",
        "email": "super_admin@genesis.ai",
        "role": "super_admin",
        "name": "Usuario Super Administrador"
    }
}

def auth_status():
    """Verificar estado de autenticación del usuario."""
    user_data = session.get('user')
    
    if user_data:
        # Eliminar datos sensibles
        safe_user = {k: v for k, v in user_data.items() if k != 'password'}
        return jsonify({
            "authenticated": True,
            "user": safe_user
        })
    
    return jsonify({
        "authenticated": False
    })

def auth_login():
    """Iniciar sesión."""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({
            "success": False,
            "message": "Usuario y contraseña son requeridos"
        }), 400
    
    user = DEMO_USERS.get(username)
    
    if user and user['password'] == password:
        # Crear sesión
        session['user'] = user
        
        # Eliminar datos sensibles
        safe_user = {k: v for k, v in user.items() if k != 'password'}
        
        return jsonify({
            "success": True,
            "user": safe_user
        })
    
    return jsonify({
        "success": False,
        "message": "Credenciales incorrectas"
    }), 401

def auth_logout():
    """Cerrar sesión."""
    session.pop('user', None)
    return jsonify({
        "success": True,
        "message": "Sesión cerrada correctamente"
    })

def register_auth_routes(app):
    """Registrar rutas de autenticación en la aplicación Flask."""
    app.add_url_rule('/api/auth/status', 'auth_status', auth_status)
    app.add_url_rule('/api/auth/login', 'auth_login', auth_login, methods=['POST'])
    app.add_url_rule('/api/auth/logout', 'auth_logout', auth_logout, methods=['POST'])