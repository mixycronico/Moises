import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { FaUsers, FaChartLine, FaCog, FaSignOutAlt } from 'react-icons/fa';

// Componentes (se crearán después)
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Loading from '../components/Loading';

const AdminDashboard = ({ user }) => {
  const [loading, setLoading] = useState(true);
  const [investors, setInvestors] = useState([]);
  const [systemStatus, setSystemStatus] = useState({});
  const [activeSection, setActiveSection] = useState('overview');

  useEffect(() => {
    const fetchData = async () => {
      try {
        // En una implementación real, estas serían llamadas separadas
        const investorsResponse = await axios.get('/api/admin/investors', { withCredentials: true });
        const statusResponse = await axios.get('/api/admin/system-status', { withCredentials: true });
        
        setInvestors(investorsResponse.data);
        setSystemStatus(statusResponse.data);
      } catch (error) {
        console.error('Error al cargar datos de administrador:', error);
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
    <div className="dashboard-container admin-dashboard">
      <Navbar user={user} onLogout={handleLogout} />
      
      <div className="dashboard-content">
        <Sidebar 
          activeSection={activeSection}
          setActiveSection={setActiveSection}
          role="admin"
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
                <h2>Panel de Control</h2>
                <div className="stats-container">
                  {/* Aquí irían los componentes de estadísticas */}
                </div>
                <div className="system-summary">
                  {/* Aquí iría el resumen del sistema */}
                </div>
              </div>
            )}
            
            {activeSection === 'investors' && (
              <div className="investors-section">
                <h2>Gestión de Inversores</h2>
                {/* Aquí iría la tabla de inversores */}
              </div>
            )}
            
            {activeSection === 'reports' && (
              <div className="reports-section">
                <h2>Informes y Análisis</h2>
                {/* Aquí irían los informes */}
              </div>
            )}
            
            {activeSection === 'settings' && (
              <div className="settings-section">
                <h2>Configuración del Sistema</h2>
                {/* Aquí irían las opciones de configuración */}
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

export default AdminDashboard;