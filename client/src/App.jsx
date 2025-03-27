import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './utils/AuthContext';

// Layouts
import MainLayout from './components/MainLayout';

// Pages
import Home from './pages/Home';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import NotFound from './pages/NotFound';

// CSS
import './styles/app.css';

// Componente para rutas protegidas
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, isLoading, user } = useAuth();
  
  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-cosmic-dark">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cosmic-accent"></div>
      </div>
    );
  }
  
  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }
  
  return children;
};

// Componente para rutas públicas (redirigen al dashboard si ya está autenticado)
const PublicRoute = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();
  
  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-cosmic-dark">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cosmic-accent"></div>
      </div>
    );
  }
  
  if (isAuthenticated) {
    return <Navigate to="/dashboard" />;
  }
  
  return children;
};

const AppRoutes = () => {
  const { user } = useAuth();

  return (
    <Router>
      <Routes>
        {/* Rutas públicas */}
        <Route path="/" element={<PublicRoute><Home /></PublicRoute>} />
        <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
        
        {/* Rutas protegidas */}
        <Route path="/" element={<ProtectedRoute><MainLayout user={user} /></ProtectedRoute>}>
          <Route path="/dashboard" element={<Dashboard />} />
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

// Componente principal que envuelve toda la aplicación con el proveedor de autenticación
const App = () => {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
};

export default App;