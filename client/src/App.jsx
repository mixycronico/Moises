import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

// Layouts
import MainLayout from './components/MainLayout';

// Pages
import Home from './pages/Home';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import NotFound from './pages/NotFound';

// CSS
import './styles/app.css';

const App = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Verificar sesión
  useEffect(() => {
    const checkSession = async () => {
      try {
        const response = await axios.get('/api/auth/status');
        if (response.data.authenticated) {
          setUser(response.data.user);
        }
      } catch (error) {
        console.error('Error checking session:', error);
      } finally {
        setLoading(false);
      }
    };
    
    checkSession();
  }, []);
  
  // Rutas protegidas
  const ProtectedRoute = ({ children }) => {
    if (loading) {
      return (
        <div className="h-screen w-screen flex items-center justify-center bg-cosmic-dark">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cosmic-accent"></div>
        </div>
      );
    }
    
    if (!user) {
      return <Navigate to="/login" />;
    }
    
    return children;
  };
  
  // Rutas públicas (redirigen al dashboard si ya está autenticado)
  const PublicRoute = ({ children }) => {
    if (loading) {
      return (
        <div className="h-screen w-screen flex items-center justify-center bg-cosmic-dark">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cosmic-accent"></div>
        </div>
      );
    }
    
    if (user) {
      return <Navigate to="/dashboard" />;
    }
    
    return children;
  };

  return (
    <Router>
      <Routes>
        {/* Rutas públicas */}
        <Route path="/" element={<PublicRoute><Home /></PublicRoute>} />
        <Route path="/login" element={<PublicRoute><Login setUser={setUser} /></PublicRoute>} />
        
        {/* Rutas protegidas */}
        <Route path="/" element={<ProtectedRoute><MainLayout user={user} /></ProtectedRoute>}>
          <Route path="/dashboard" element={<Dashboard user={user} />} />
          <Route path="/trading" element={<div className="p-6">Página de Trading (En desarrollo)</div>} />
          <Route path="/investments" element={<div className="p-6">Página de Inversiones (En desarrollo)</div>} />
          <Route path="/history" element={<div className="p-6">Página de Historial (En desarrollo)</div>} />
          <Route path="/performance" element={<div className="p-6">Página de Rendimiento (En desarrollo)</div>} />
          
          {/* Rutas de administrador */}
          <Route path="/investors" element={<div className="p-6">Administración de Inversionistas (En desarrollo)</div>} />
          <Route path="/commissions" element={<div className="p-6">Administración de Comisiones (En desarrollo)</div>} />
          
          {/* Rutas de super administrador */}
          <Route path="/system" element={<div className="p-6">Configuración del Sistema (En desarrollo)</div>} />
          <Route path="/database" element={<div className="p-6">Administración de Base de Datos (En desarrollo)</div>} />
          
          {/* Rutas de configuración */}
          <Route path="/settings" element={<div className="p-6">Configuración de Usuario (En desarrollo)</div>} />
          <Route path="/help" element={<div className="p-6">Ayuda y Soporte (En desarrollo)</div>} />
          <Route path="/profile" element={<div className="p-6">Perfil de Usuario (En desarrollo)</div>} />
        </Route>
        
        {/* Ruta 404 */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
};

export default App;