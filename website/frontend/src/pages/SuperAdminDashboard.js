import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { FaUsers, FaChartLine, FaCog, FaServer, FaDatabase, FaSignOutAlt } from 'react-icons/fa';

// Componentes (se crearán después)
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Loading from '../components/Loading';

const SuperAdminDashboard = ({ user }) => {
  const [loading, setLoading] = useState(true);
  const [systemStats, setSystemStats] = useState({});
  const [adminUsers, setAdminUsers] = useState([]);
  const [activeSection, setActiveSection] = useState('overview');

  useEffect(() => {
    const fetchData = async () => {
      try {
        // En una implementación real, estas serían llamadas separadas
        const statsResponse = await axios.get('/api/super-admin/system-stats', { withCredentials: true });
        const adminsResponse = await axios.get('/api/super-admin/admins', { withCredentials: true });
        
        setSystemStats(statsResponse.data);
        setAdminUsers(adminsResponse.data);
      } catch (error) {
        console.error('Error al cargar datos de super administrador:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleLogout = async () => {
    try {
      await axios.post('/api/logout', {}, { withCredentials: true });
      window.location.href = '/login';
    } catch (error) {
      console.error('Error al cerrar sesión:', error);
    }
  };

  if (loading) {
    return <Loading />;
  }

  return (
    <div className="dashboard-container super-admin-dashboard">
      <Navbar user={user} onLogout={handleLogout} />
      
      <div className="dashboard-content">
        <Sidebar 
          activeSection={activeSection}
          setActiveSection={setActiveSection}
          role="super_admin"
        />
        
        <main className="dashboard-main">
          <motion.div
            key={activeSection}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="dashboard-section"
          >
            {activeSection === 'overview' && (
              <div className="overview-section">
                <h2>Centro de Control</h2>
                <div className="stats-container">
                  {/* Aquí irían los componentes de estadísticas avanzadas */}
                </div>
                <div className="system-health">
                  {/* Aquí iría el resumen de salud del sistema */}
                </div>
              </div>
            )}
            
            {activeSection === 'admins' && (
              <div className="admins-section">
                <h2>Gestión de Administradores</h2>
                {/* Aquí iría la tabla de administradores */}
              </div>
            )}
            
            {activeSection === 'system' && (
              <div className="system-section">
                <h2>Configuración del Sistema</h2>
                {/* Aquí irían las opciones avanzadas del sistema */}
              </div>
            )}
            
            {activeSection === 'database' && (
              <div className="database-section">
                <h2>Gestión de Base de Datos</h2>
                {/* Aquí irían las opciones de base de datos */}
              </div>
            )}
            
            {activeSection === 'logs' && (
              <div className="logs-section">
                <h2>Registros del Sistema</h2>
                {/* Aquí irían los logs del sistema */}
              </div>
            )}
          </motion.div>
        </main>
      </div>
      
      <div className="cosmic-background dashboard-background">
        {/* Elementos visuales de fondo (estrellas, nebulosas, etc.) */}
      </div>
    </div>
  );
};

export default SuperAdminDashboard;