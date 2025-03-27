import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  FiHome, FiTrendingUp, FiDollarSign, FiUsers, 
  FiActivity, FiSettings, FiHelpCircle, FiClock,
  FiLayers, FiShield, FiDatabase
} from 'react-icons/fi';

const Sidebar = ({ open, user }) => {
  const userRole = user?.role || 'user';
  
  // Menú básico para todos los usuarios
  const baseMenu = [
    { 
      path: '/dashboard', 
      label: 'Dashboard', 
      icon: <FiHome />,
      roles: ['user', 'admin', 'super_admin', 'creator'] 
    },
    { 
      path: '/trading', 
      label: 'Trading', 
      icon: <FiTrendingUp />,
      roles: ['user', 'admin', 'super_admin', 'creator'] 
    },
    { 
      path: '/investments', 
      label: 'Inversiones', 
      icon: <FiDollarSign />,
      roles: ['user', 'admin', 'super_admin', 'creator'] 
    },
    { 
      path: '/history', 
      label: 'Historial', 
      icon: <FiClock />,
      roles: ['user', 'admin', 'super_admin', 'creator'] 
    },
    { 
      path: '/performance', 
      label: 'Rendimiento', 
      icon: <FiActivity />,
      roles: ['user', 'admin', 'super_admin', 'creator'] 
    },
  ];
  
  // Menú para administradores
  const adminMenu = [
    { 
      path: '/investors', 
      label: 'Inversionistas', 
      icon: <FiUsers />,
      roles: ['admin', 'super_admin', 'creator'] 
    },
    { 
      path: '/commissions', 
      label: 'Comisiones', 
      icon: <FiLayers />,
      roles: ['admin', 'super_admin', 'creator'] 
    }
  ];
  
  // Menú para super administradores
  const superAdminMenu = [
    { 
      path: '/system', 
      label: 'Sistema', 
      icon: <FiShield />,
      roles: ['super_admin', 'creator'] 
    },
    { 
      path: '/database', 
      label: 'Base de Datos', 
      icon: <FiDatabase />,
      roles: ['super_admin', 'creator'] 
    }
  ];
  
  // Menú de configuración para todos
  const settingsMenu = [
    { 
      path: '/settings', 
      label: 'Configuración', 
      icon: <FiSettings />,
      roles: ['user', 'admin', 'super_admin', 'creator'] 
    },
    { 
      path: '/help', 
      label: 'Ayuda', 
      icon: <FiHelpCircle />,
      roles: ['user', 'admin', 'super_admin', 'creator'] 
    }
  ];
  
  // Filtrar menú según rol del usuario
  const filterMenuByRole = (menuItems) => {
    return menuItems.filter(item => item.roles.includes(userRole));
  };
  
  // Obtener todos los elementos del menú filtrados por rol
  const allMenuItems = [
    ...filterMenuByRole(baseMenu),
    ...filterMenuByRole(adminMenu),
    ...filterMenuByRole(superAdminMenu)
  ];
  
  const settingsItems = filterMenuByRole(settingsMenu);
  
  // Variantes para animaciones
  const sidebarVariants = {
    open: {
      width: '240px',
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 24
      }
    },
    closed: {
      width: '68px',
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 24
      }
    }
  };
  
  const menuLabelVariants = {
    open: {
      opacity: 1,
      x: 0,
      display: 'block',
      transition: {
        delay: 0.1,
        duration: 0.2
      }
    },
    closed: {
      opacity: 0,
      x: -10,
      transitionEnd: {
        display: 'none'
      },
      transition: {
        duration: 0.2
      }
    }
  };

  return (
    <motion.nav
      className="h-screen fixed top-0 left-0 bg-cosmic-primary/20 backdrop-blur-md border-r border-cosmic-primary/30 flex flex-col z-20"
      initial={open ? 'open' : 'closed'}
      animate={open ? 'open' : 'closed'}
      variants={sidebarVariants}
    >
      {/* Logo */}
      <div className={`h-16 flex items-center px-4 border-b border-cosmic-primary/30 ${!open && 'justify-center'}`}>
        <div className="w-8 h-8 bg-cosmic-gradient rounded-full flex items-center justify-center">
          <span className="text-white font-bold">G</span>
        </div>
        <motion.span 
          className="ml-2 text-lg font-semibold cosmic-gradient-text overflow-hidden whitespace-nowrap"
          variants={menuLabelVariants}
        >
          Genesis
        </motion.span>
      </div>
      
      {/* Categoría de usuario */}
      <div className="mt-4 px-4 mb-6">
        <div className={`py-2 px-3 bg-cosmic-primary/20 rounded-md border border-cosmic-primary/30 ${!open ? 'justify-center' : ''} flex items-center`}>
          <div className="w-6 h-6 rounded-full bg-cosmic-highlight/30 flex items-center justify-center text-cosmic-highlight text-xs">
            {userRole === 'admin' ? 'A' : userRole === 'super_admin' ? 'S' : userRole === 'creator' ? 'C' : 'U'}
          </div>
          <motion.div 
            className="ml-2 overflow-hidden whitespace-nowrap"
            variants={menuLabelVariants}
          >
            <p className="text-sm font-medium capitalize text-white">
              {userRole === 'super_admin' ? 'Super Admin' : userRole}
            </p>
            <p className="text-xs text-cosmic-glow uppercase">
              {user?.category || 'silver'}
            </p>
          </motion.div>
        </div>
      </div>
      
      {/* Main Menu */}
      <div className="flex-1 px-2 overflow-y-auto">
        <ul className="space-y-1">
          {allMenuItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) => `
                  flex items-center py-2 px-3 rounded-md
                  ${isActive 
                    ? 'bg-cosmic-primary/40 text-white font-medium' 
                    : 'hover:bg-cosmic-primary/20 text-gray-300'
                  }
                  transition-colors
                `}
              >
                <span className="text-xl">{item.icon}</span>
                <motion.span 
                  className="ml-3 overflow-hidden whitespace-nowrap"
                  variants={menuLabelVariants}
                >
                  {item.label}
                </motion.span>
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
      
      {/* Settings Menu */}
      <div className="px-2 py-4 border-t border-cosmic-primary/30">
        <ul className="space-y-1">
          {settingsItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) => `
                  flex items-center py-2 px-3 rounded-md
                  ${isActive 
                    ? 'bg-cosmic-primary/40 text-white font-medium' 
                    : 'hover:bg-cosmic-primary/20 text-gray-300'
                  }
                  transition-colors
                `}
              >
                <span className="text-xl">{item.icon}</span>
                <motion.span 
                  className="ml-3 overflow-hidden whitespace-nowrap"
                  variants={menuLabelVariants}
                >
                  {item.label}
                </motion.span>
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
      
      {/* Version number */}
      <div className="px-4 py-2 text-xs text-gray-500 text-center">
        <motion.p variants={menuLabelVariants}>
          v4.4 Quantum Ultra Divino
        </motion.p>
      </div>
    </motion.nav>
  );
};

export default Sidebar;