import React from 'react';
import { motion } from 'framer-motion';
import { 
  FaChartLine, FaWallet, FaHistory, FaCog, 
  FaUsers, FaServer, FaDatabase, FaFileAlt, 
  FaTachometerAlt, FaShieldAlt
} from 'react-icons/fa';

const Sidebar = ({ activeSection, setActiveSection, role }) => {
  // Definir menús para cada rol
  const menuItems = {
    investor: [
      { id: 'overview', label: 'Resumen', icon: <FaTachometerAlt /> },
      { id: 'portfolio', label: 'Portafolio', icon: <FaWallet /> },
      { id: 'transactions', label: 'Transacciones', icon: <FaHistory /> },
      { id: 'settings', label: 'Configuración', icon: <FaCog /> }
    ],
    admin: [
      { id: 'overview', label: 'Panel de Control', icon: <FaTachometerAlt /> },
      { id: 'investors', label: 'Inversores', icon: <FaUsers /> },
      { id: 'reports', label: 'Informes', icon: <FaFileAlt /> },
      { id: 'settings', label: 'Configuración', icon: <FaCog /> }
    ],
    super_admin: [
      { id: 'overview', label: 'Centro de Control', icon: <FaTachometerAlt /> },
      { id: 'admins', label: 'Administradores', icon: <FaUsers /> },
      { id: 'system', label: 'Sistema', icon: <FaServer /> },
      { id: 'database', label: 'Base de Datos', icon: <FaDatabase /> },
      { id: 'logs', label: 'Registros', icon: <FaFileAlt /> },
      { id: 'security', label: 'Seguridad', icon: <FaShieldAlt /> }
    ]
  };

  // Seleccionar el menú correcto según el rol
  const items = menuItems[role] || menuItems.investor;

  return (
    <div className="sidebar">
      <div className="sidebar-items">
        {items.map((item) => (
          <motion.div
            key={item.id}
            className={`sidebar-item ${activeSection === item.id ? 'active' : ''}`}
            onClick={() => setActiveSection(item.id)}
            whileHover={{ 
              scale: 1.05,
              backgroundColor: 'rgba(255, 255, 255, 0.1)' 
            }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="sidebar-icon">{item.icon}</div>
            <span className="sidebar-label">{item.label}</span>
            {activeSection === item.id && (
              <motion.div 
                className="active-indicator"
                layoutId="activeIndicator"
                transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              />
            )}
          </motion.div>
        ))}
      </div>
      
      <div className="sidebar-footer">
        <div className="system-status">
          <div className="status-indicator online"></div>
          <span>Sistema Activo</span>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;