import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// Páginas
import Login from './pages/Login';
import Index from './pages/Index';
import InvestorDashboard from './pages/InvestorDashboard';
import AdminDashboard from './pages/AdminDashboard';
import SuperAdminDashboard from './pages/SuperAdminDashboard';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Verificar si el usuario está autenticado al cargar la aplicación
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await axios.get('/api/check-auth', { withCredentials: true });
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

  // Rutas protegidas basadas en el rol del usuario
  const ProtectedRoute = ({ children, allowedRoles }) => {
    if (loading) return <div>Cargando...</div>;
    
    if (!user) {
      return <Navigate to="/login" />;
    }
    
    if (allowedRoles.includes(user.role)) {
      return children;
    }
    
    // Redirigir a la página apropiada según el rol
    if (user.role === 'super_admin') {
      return <Navigate to="/super-admin" />;
    } else if (user.role === 'admin') {
      return <Navigate to="/admin" />;
    } else {
      return <Navigate to="/investor" />;
    }
  };

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Index />} />
        <Route path="/login" element={<Login setUser={setUser} />} />
        
        <Route 
          path="/investor" 
          element={
            <ProtectedRoute allowedRoles={['investor', 'admin', 'super_admin']}>
              <InvestorDashboard user={user} />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/admin" 
          element={
            <ProtectedRoute allowedRoles={['admin', 'super_admin']}>
              <AdminDashboard user={user} />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/super-admin" 
          element={
            <ProtectedRoute allowedRoles={['super_admin']}>
              <SuperAdminDashboard user={user} />
            </ProtectedRoute>
          } 
        />
        
        {/* Redirección por defecto basada en el rol del usuario */}
        <Route 
          path="/dashboard" 
          element={
            user ? (
              user.role === 'super_admin' ? (
                <Navigate to="/super-admin" />
              ) : user.role === 'admin' ? (
                <Navigate to="/admin" />
              ) : (
                <Navigate to="/investor" />
              )
            ) : (
              <Navigate to="/login" />
            )
          } 
        />
      </Routes>
    </Router>
  );
}

export default App;