import React, { createContext, useState, useContext, useEffect } from 'react';
import { checkAuthStatus, login as apiLogin, logout as apiLogout } from '../services/api';

// Crear contexto
const AuthContext = createContext();

// Hook personalizado para usar el contexto
export const useAuth = () => useContext(AuthContext);

// Proveedor de contexto
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Verificar estado de autenticación al cargar la aplicación
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const { authenticated, user } = await checkAuthStatus();
        setIsAuthenticated(authenticated);
        setUser(user);
      } catch (err) {
        console.error('Error al verificar autenticación:', err);
        setIsAuthenticated(false);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  // Función para iniciar sesión
  const login = async (credentials) => {
    setIsLoading(true);
    setError(null);
    try {
      const { success, user, token } = await apiLogin(credentials);
      if (success && token) {
        localStorage.setItem('token', token);
        setUser(user);
        setIsAuthenticated(true);
        return { success: true };
      } else {
        throw new Error('Credenciales inválidas');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Error al iniciar sesión');
      return { success: false, error: err.response?.data?.message || 'Error al iniciar sesión' };
    } finally {
      setIsLoading(false);
    }
  };

  // Función para cerrar sesión
  const logout = async () => {
    setIsLoading(true);
    try {
      await apiLogout();
    } catch (err) {
      console.error('Error al cerrar sesión:', err);
    } finally {
      localStorage.removeItem('token');
      setUser(null);
      setIsAuthenticated(false);
      setIsLoading(false);
    }
  };

  // Valor del contexto
  const value = {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthContext;