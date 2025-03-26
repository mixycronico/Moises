import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

/**
 * Contexto de autenticación para gestionar el inicio de sesión y autorización
 */
const AuthContext = createContext(null);

/**
 * Proveedor del contexto de autenticación
 */
export const AuthProvider = ({ children }) => {
  // Estado para el usuario
  const [user, setUser] = useState(null);
  // Estado de carga
  const [loading, setLoading] = useState(true);

  // Verificar si el usuario ya está autenticado al cargar la aplicación
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        const response = await axios.get('/api/auth/status', {
          withCredentials: true,
        });
        
        if (response.data.authenticated) {
          setUser(response.data.user);
        }
        
        setLoading(false);
      } catch (error) {
        console.error('Error al verificar estado de autenticación:', error);
        setLoading(false);
      }
    };

    checkAuthStatus();
  }, []);

  /**
   * Iniciar sesión con nombre de usuario y contraseña
   */
  const login = async (username, password) => {
    try {
      const response = await axios.post('/api/auth/login', {
        username,
        password,
      }, {
        withCredentials: true,
      });

      setUser(response.data.user);
      return { success: true, user: response.data.user };
    } catch (error) {
      console.error('Error de inicio de sesión:', error);
      let errorMessage = 'Error al iniciar sesión';
      
      if (error.response && error.response.data && error.response.data.message) {
        errorMessage = error.response.data.message;
      }
      
      return { success: false, error: errorMessage };
    }
  };

  /**
   * Cerrar sesión
   */
  const logout = async () => {
    try {
      await axios.post('/api/auth/logout', {}, {
        withCredentials: true,
      });
      
      setUser(null);
      return { success: true };
    } catch (error) {
      console.error('Error al cerrar sesión:', error);
      return { success: false, error: 'Error al cerrar sesión' };
    }
  };

  /**
   * Determinar si el usuario tiene un rol específico
   */
  const hasRole = (role) => {
    if (!user) return false;
    return user.role === role;
  };
  
  /**
   * Valores y funciones expuestas por el contexto
   */
  const value = {
    user,
    loading,
    login,
    logout,
    hasRole,
    isAuthenticated: !!user,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

/**
 * Hook personalizado para acceder al contexto de autenticación
 */
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth debe ser utilizado dentro de un AuthProvider');
  }
  return context;
};

export default AuthContext;