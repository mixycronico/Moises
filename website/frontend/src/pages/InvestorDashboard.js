import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { FaChartLine, FaWallet, FaHistory, FaSignOutAlt } from 'react-icons/fa';

// Componentes (se crearán después)
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Loading from '../components/Loading';

const InvestorDashboard = ({ user }) => {
  const [loading, setLoading] = useState(true);
  const [portfolioData, setPortfolioData] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [activeSection, setActiveSection] = useState('overview');

  useEffect(() => {
    const fetchData = async () => {
      try {
        // En una implementación real, estas serían llamadas separadas
        const portfolioResponse = await axios.get('/api/investor/portfolio', { withCredentials: true });
        const transactionsResponse = await axios.get('/api/investor/transactions', { withCredentials: true });
        
        setPortfolioData(portfolioResponse.data);
        setTransactions(transactionsResponse.data);
      } catch (error) {
        console.error('Error al cargar datos del inversor:', error);
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
    <div className="dashboard-container investor-dashboard">
      <Navbar user={user} onLogout={handleLogout} />
      
      <div className="dashboard-content">
        <Sidebar 
          activeSection={activeSection}
          setActiveSection={setActiveSection}
          role="investor"
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
                <h2>Resumen de Inversiones</h2>
                <div className="stats-container">
                  {/* Aquí irían los componentes de estadísticas */}
                </div>
                <div className="portfolio-summary">
                  {/* Aquí iría el resumen del portafolio */}
                </div>
              </div>
            )}
            
            {activeSection === 'portfolio' && (
              <div className="portfolio-section">
                <h2>Portafolio Detallado</h2>
                {/* Aquí iría la tabla o gráficos del portafolio */}
              </div>
            )}
            
            {activeSection === 'transactions' && (
              <div className="transactions-section">
                <h2>Historial de Transacciones</h2>
                {/* Aquí iría la tabla de transacciones */}
              </div>
            )}
            
            {activeSection === 'settings' && (
              <div className="settings-section">
                <h2>Configuración</h2>
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

export default InvestorDashboard;