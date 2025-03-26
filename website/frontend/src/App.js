import React, { useState, useEffect } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';

// Páginas principales
import Index from './pages/Index';
import Login from './pages/Login';
import NotFound from './pages/NotFound';

// Dashboards según rol
import InvestorDashboard from './pages/InvestorDashboard';
import AdminDashboard from './pages/AdminDashboard';
import SuperAdminDashboard from './pages/SuperAdminDashboard';

// Rutas protegidas y servicio de autenticación
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider, useAuth } from './utils/AuthContext';

// Componentes comunes
import Loading from './components/Loading';
import NavBar from './components/NavBar';

const AppContent = () => {
  const [loading, setLoading] = useState(true);
  const { user, checkAuth } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Verificar estado de autenticación al cargar
  useEffect(() => {
    const verifyAuth = async () => {
      try {
        await checkAuth();
      } catch (error) {
        console.error('Error verificando autenticación:', error);
      } finally {
        setLoading(false);
      }
    };

    verifyAuth();
  }, [checkAuth]);

  // Renderizar loader mientras se verifica autenticación
  if (loading) {
    return <Loading />;
  }

  return (
    <>
      {/* Mostrar navbar solo si no estamos en la página de login o index */}
      {!['/login', '/'].includes(location.pathname) && <NavBar />}

      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          {/* Rutas públicas */}
          <Route path="/" element={<Index />} />
          <Route path="/login" element={<Login />} />

          {/* Rutas de inversor */}
          <Route 
            path="/investor/*" 
            element={
              <ProtectedRoute allowedRoles={['investor']}>
                <Routes>
                  <Route path="/" element={<InvestorDashboard />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </ProtectedRoute>
            } 
          />

          {/* Rutas de administrador */}
          <Route 
            path="/admin/*" 
            element={
              <ProtectedRoute allowedRoles={['admin', 'super_admin']}>
                <Routes>
                  <Route path="/" element={<AdminDashboard />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </ProtectedRoute>
            } 
          />

          {/* Rutas de super administrador */}
          <Route 
            path="/super-admin/*" 
            element={
              <ProtectedRoute allowedRoles={['super_admin']}>
                <Routes>
                  <Route path="/" element={<SuperAdminDashboard />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </ProtectedRoute>
            } 
          />

          {/* Ruta 404 */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AnimatePresence>
    </>
  );
};

const App = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;