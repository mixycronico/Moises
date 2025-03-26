import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../utils/AuthContext';
import Loading from './Loading';

/**
 * Componente para proteger rutas y controlar acceso basado en roles
 * 
 * @param {Object} props - Propiedades del componente
 * @param {React.ReactNode} props.children - Componente hijo a renderizar si el usuario está autorizado
 * @param {string[]} [props.allowedRoles] - Roles permitidos para acceder a la ruta
 * @param {boolean} [props.requireAuth=true] - Si se requiere autenticación
 * @returns {React.ReactNode}
 */
const ProtectedRoute = ({
  children,
  allowedRoles = [],
  requireAuth = true,
}) => {
  const { isAuthenticated, loading, user } = useAuth();
  const location = useLocation();

  // Mostrar pantalla de carga mientras se verifica la autenticación
  if (loading) {
    return <Loading message="Verificando sesión..." />;
  }

  // Si no se requiere autenticación pero el usuario está autenticado
  if (!requireAuth && isAuthenticated) {
    // Redirigir a la página correspondiente según el rol
    if (user.role === 'super_admin') {
      return <Navigate to="/super-admin" state={{ from: location }} replace />;
    } else if (user.role === 'admin') {
      return <Navigate to="/admin" state={{ from: location }} replace />;
    } else if (user.role === 'investor') {
      return <Navigate to="/investor" state={{ from: location }} replace />;
    }
  }

  // Si no se requiere autenticación y el usuario no está autenticado, mostrar el componente
  if (!requireAuth && !isAuthenticated) {
    return children;
  }

  // Si se requiere autenticación pero el usuario no está autenticado
  if (requireAuth && !isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Si hay roles permitidos y el usuario no tiene ninguno de esos roles
  if (allowedRoles.length > 0 && !allowedRoles.includes(user.role)) {
    // Redirigir a la página correspondiente según el rol
    if (user.role === 'super_admin') {
      return <Navigate to="/super-admin" state={{ from: location }} replace />;
    } else if (user.role === 'admin') {
      return <Navigate to="/admin" state={{ from: location }} replace />;
    } else if (user.role === 'investor') {
      return <Navigate to="/investor" state={{ from: location }} replace />;
    } else {
      // Si el rol no se reconoce, redirigir a página no autorizada
      return <Navigate to="/unauthorized" state={{ from: location }} replace />;
    }
  }

  // Si todo está bien, mostrar el componente hijo
  return children;
};

export default ProtectedRoute;