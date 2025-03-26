import React, { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../utils/AuthContext';
import Loading from './Loading';

/**
 * Componente para proteger rutas basado en roles
 * @param {Object} props - Propiedades del componente
 * @param {Array} props.allowedRoles - Roles permitidos para acceder a la ruta
 * @param {React.ReactNode} props.children - Componentes hijos a renderizar si el usuario está autorizado
 * @param {string} [props.redirectPath='/login'] - Ruta a la que redirigir si el usuario no está autorizado
 */
const ProtectedRoute = ({ 
  allowedRoles, 
  children, 
  redirectPath = '/login' 
}) => {
  const { user, loading, hasRole, checkAuth } = useAuth();
  const location = useLocation();

  // Verificar autenticación al cargar el componente
  useEffect(() => {
    if (!user) {
      checkAuth();
    }
  }, [checkAuth, user]);

  // Mientras verifica la autenticación, mostrar indicador de carga
  if (loading) {
    return <Loading />;
  }

  // Si no hay usuario autenticado, redirigir al login
  if (!user) {
    return <Navigate to={redirectPath} state={{ from: location }} replace />;
  }

  // Verificar si el usuario tiene los roles permitidos
  const isAuthorized = hasRole(allowedRoles);

  // Si el usuario no tiene los roles necesarios, redirigir
  if (!isAuthorized) {
    // Determinar a dónde redirigir basado en el rol del usuario
    let redirectTo = '/login';
    
    if (user.role === 'investor') {
      redirectTo = '/investor';
    } else if (user.role === 'admin') {
      redirectTo = '/admin';
    } else if (user.role === 'super_admin') {
      redirectTo = '/super-admin';
    }
    
    return <Navigate to={redirectTo} replace />;
  }

  // Si está autorizado, renderizar los hijos
  return children;
};

export default ProtectedRoute;