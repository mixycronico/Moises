import React, { createContext, useState, useContext, useCallback } from 'react';
import axios from 'axios';

// Crear contexto de autenticación
const AuthContext = createContext(null);

// URL base para API
const API_URL = '/api';

export const AuthProvider = ({ children }) => {
  // Estado para almacenar información del usuario
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Comprobar estado de autenticación
  const checkAuth = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_URL}/check-auth`, { withCredentials: true });
      
      if (response.data.authenticated) {
        setUser(response.data.user);
      } else {
        setUser(null);
      }
      
      return response.data.authenticated;
    } catch (err) {
      console.error('Error al verificar autenticación:', err);
      setUser(null);
      setError('Error al verificar autenticación');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  // Iniciar sesión
  const login = useCallback(async (username, password) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_URL}/login`, {
        username,
        password
      }, { withCredentials: true });
      
      if (response.data.success) {
        setUser(response.data.user);
        return { success: true, user: response.data.user };
      } else {
        setError(response.data.message || 'Error al iniciar sesión');
        return { success: false, message: response.data.message };
      }
    } catch (err) {
      console.error('Error al iniciar sesión:', err);
      
      let errorMessage = 'Error al iniciar sesión';
      if (err.response && err.response.data && err.response.data.message) {
        errorMessage = err.response.data.message;
      }
      
      setError(errorMessage);
      return { success: false, message: errorMessage };
    } finally {
      setLoading(false);
    }
  }, []);

  // Cerrar sesión
  const logout = useCallback(async () => {
    setLoading(true);
    
    try {
      await axios.post(`${API_URL}/logout`, {}, { withCredentials: true });
      setUser(null);
      return true;
    } catch (err) {
      console.error('Error al cerrar sesión:', err);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  // Verificar si el usuario tiene un rol específico
  const hasRole = useCallback((roles) => {
    if (!user) return false;
    
    if (Array.isArray(roles)) {
      return roles.includes(user.role);
    }
    
    return user.role === roles;
  }, [user]);

  // Valores que estarán disponibles en el contexto
  const contextValue = {
    user,
    loading,
    error,
    checkAuth,
    login,
    logout,
    hasRole
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

// Hook personalizado para usar el contexto de autenticación
export const useAuth = () => {
  const context = useContext(AuthContext);
  
  if (!context) {
    throw new Error('useAuth debe ser usado dentro de un AuthProvider');
  }
  
  return context;
};