import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './utils/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import NavBar from './components/NavBar';
import Index from './pages/Index';
import Login from './pages/Login';
import NotFound from './pages/NotFound';
import InvestorDashboard from './pages/InvestorDashboard';
import AdminDashboard from './pages/AdminDashboard';
import SuperAdminDashboard from './pages/SuperAdminDashboard';

/**
 * Componente principal de la aplicación
 */
function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="app">
          <NavBar />
          <Routes>
            {/* Rutas públicas */}
            <Route path="/" element={<Index />} />
            <Route 
              path="/login" 
              element={
                <ProtectedRoute requireAuth={false}>
                  <Login />
                </ProtectedRoute>
              } 
            />
            
            {/* Ruta de inversor */}
            <Route 
              path="/investor" 
              element={
                <ProtectedRoute allowedRoles={['investor']}>
                  <InvestorDashboard />
                </ProtectedRoute>
              } 
            />
            
            {/* Ruta de admin */}
            <Route 
              path="/admin" 
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminDashboard />
                </ProtectedRoute>
              } 
            />
            
            {/* Ruta de super admin */}
            <Route 
              path="/super-admin" 
              element={
                <ProtectedRoute allowedRoles={['super_admin']}>
                  <SuperAdminDashboard />
                </ProtectedRoute>
              } 
            />
            
            {/* Ruta 404 */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  );
}

// Estilos globales
const globalStyles = `
  :root {
    /* Colores */
    --color-primary: #0cc6de;
    --color-secondary: #9270ff;
    --color-accent: #ff27b3;
    --color-background: #0d1930;
    --color-card-bg: rgba(22, 43, 77, 0.7);
    --color-text: #e1eaff;
    --color-text-secondary: #8b9ebb;
    --color-border: rgba(70, 137, 180, 0.3);
    --color-success: #3effa3;
    --color-warning: #ffb930;
    --color-danger: #ff5371;
    
    /* Espaciados */
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2.5rem;
    
    /* Bordes */
    --border-radius-sm: 4px;
    --border-radius-md: 8px;
    --border-radius-lg: 12px;
    --border-radius-circle: 50%;
    
    /* Tipografía */
    --font-display: 'Orbitron', sans-serif;
    --font-secondary: 'Exo 2', sans-serif;
    --font-body: 'Inter', sans-serif;
    
    /* Transiciones */
    --transition-fast: 0.2s ease;
    --transition-normal: 0.3s ease;
    
    /* Sombras */
    --shadow-soft: 0 4px 10px rgba(0, 0, 0, 0.2);
    --shadow-medium: 0 8px 20px rgba(0, 0, 0, 0.3);
    --shadow-glow: 0 0 15px rgba(12, 198, 222, 0.5);
    
    /* Efectos especiales */
    --hologram-shimmer: linear-gradient(
      135deg,
      rgba(12, 198, 222, 0.05) 25%,
      rgba(146, 112, 255, 0.05) 50%,
      rgba(12, 198, 222, 0.05) 75%
    );
  }
  
  /* Importación de fuentes */
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  html, body {
    height: 100%;
    font-size: 16px;
  }
  
  body {
    font-family: var(--font-body);
    background-color: var(--color-background);
    color: var(--color-text);
    line-height: 1.6;
    overflow-x: hidden;
  }
  
  a {
    color: var(--color-primary);
    text-decoration: none;
    transition: color var(--transition-fast);
  }
  
  a:hover {
    color: var(--color-secondary);
  }
  
  button {
    font-family: var(--font-body);
    cursor: pointer;
  }
  
  h1, h2, h3, h4, h5, h6 {
    margin-bottom: var(--spacing-md);
    font-family: var(--font-display);
    font-weight: 600;
    line-height: 1.3;
  }
  
  h1 {
    font-size: 2.5rem;
  }
  
  h2 {
    font-size: 2rem;
  }
  
  h3 {
    font-size: 1.5rem;
  }
  
  p {
    margin-bottom: var(--spacing-md);
  }
  
  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  
  @keyframes shimmer {
    0% {
      left: -100%;
    }
    100% {
      left: 100%;
    }
  }
  
  @keyframes glow {
    0%, 100% {
      box-shadow: 0 0 15px rgba(12, 198, 222, 0.5);
    }
    50% {
      box-shadow: 0 0 25px rgba(12, 198, 222, 0.8);
    }
  }
  
  /* Responsive */
  @media (max-width: 768px) {
    html {
      font-size: 14px;
    }
    
    h1 {
      font-size: 2rem;
    }
    
    h2 {
      font-size: 1.75rem;
    }
    
    h3 {
      font-size: 1.25rem;
    }
  }
  
  @media (max-width: 480px) {
    html {
      font-size: 12px;
    }
  }
`;

// Insertar estilos globales si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'global-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = globalStyles;
    document.head.appendChild(styleElement);
  }
}

export default App;