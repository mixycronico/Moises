import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import axios from 'axios';

// Páginas
import Index from './pages/Index';
import Login from './pages/Login';
import InvestorDashboard from './pages/InvestorDashboard';
import AdminDashboard from './pages/AdminDashboard';
import SuperAdminDashboard from './pages/SuperAdminDashboard';
import NotFound from './pages/NotFound';

// Componentes
import Loading from './components/Loading';

// Configurar axios para incluir cookies en las peticiones
axios.defaults.withCredentials = true;

const App = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Verificar estado de autenticación al cargar
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await axios.get('/api/check-auth');
        if (response.data.authenticated) {
          setUser(response.data.user);
        }
      } catch (error) {
        console.error('Error al verificar autenticación:', error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  // Manejar inicio de sesión
  const handleLogin = (userData) => {
    setUser(userData);
  };

  // Manejar cierre de sesión
  const handleLogout = async () => {
    try {
      await axios.post('/api/logout');
      setUser(null);
      navigate('/login');
    } catch (error) {
      console.error('Error al cerrar sesión:', error);
    }
  };

  // Redirigir al dashboard según el rol del usuario
  const redirectToDashboard = () => {
    if (!user) return <Navigate to="/login" />;

    switch (user.role) {
      case 'super_admin':
        return <Navigate to="/super-admin" />;
      case 'admin':
        return <Navigate to="/admin" />;
      case 'investor':
        return <Navigate to="/investor" />;
      default:
        return <Navigate to="/login" />;
    }
  };

  // Rutas protegidas por rol
  const ProtectedRoute = ({ element, allowedRoles }) => {
    if (loading) return <Loading />;
    
    if (!user) {
      return <Navigate to="/login" />;
    }
    
    if (allowedRoles.includes(user.role)) {
      return element;
    }
    
    // Si el usuario no tiene el rol adecuado, redirigir según su rol
    return redirectToDashboard();
  };

  if (loading) {
    return <Loading />;
  }

  return (
    <Routes>
      {/* Rutas públicas */}
      <Route path="/" element={<Index />} />
      <Route 
        path="/login" 
        element={user ? redirectToDashboard() : <Login onLogin={handleLogin} />} 
      />

      {/* Rutas protegidas */}
      <Route 
        path="/investor/*" 
        element={
          <ProtectedRoute 
            element={<InvestorDashboard user={user} onLogout={handleLogout} />} 
            allowedRoles={['investor', 'admin', 'super_admin']} 
          />
        } 
      />
      
      <Route 
        path="/admin/*" 
        element={
          <ProtectedRoute 
            element={<AdminDashboard user={user} onLogout={handleLogout} />} 
            allowedRoles={['admin', 'super_admin']} 
          />
        } 
      />
      
      <Route 
        path="/super-admin/*" 
        element={
          <ProtectedRoute 
            element={<SuperAdminDashboard user={user} onLogout={handleLogout} />} 
            allowedRoles={['super_admin']} 
          />
        } 
      />

      {/* Ruta para dashboard (redirige según rol) */}
      <Route path="/dashboard" element={redirectToDashboard()} />
      
      {/* Manejo de rutas no encontradas */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

export default App;