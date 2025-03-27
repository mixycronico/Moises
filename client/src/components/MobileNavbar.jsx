import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  FiHome, FiTrendingUp, FiDollarSign, 
  FiActivity, FiSettings 
} from 'react-icons/fi';

const MobileNavbar = ({ user }) => {
  // Menú de navegación principal para móviles
  const navItems = [
    { 
      path: '/dashboard', 
      label: 'Dashboard', 
      icon: <FiHome size={20} />
    },
    { 
      path: '/trading', 
      label: 'Trading', 
      icon: <FiTrendingUp size={20} /> 
    },
    { 
      path: '/investments', 
      label: 'Inversiones', 
      icon: <FiDollarSign size={20} /> 
    },
    { 
      path: '/performance', 
      label: 'Rendimiento', 
      icon: <FiActivity size={20} /> 
    },
    { 
      path: '/settings', 
      label: 'Ajustes', 
      icon: <FiSettings size={20} /> 
    }
  ];

  return (
    <motion.div 
      className="fixed bottom-0 left-0 right-0 h-14 bg-cosmic-primary-20 backdrop-blur-md border-t border-cosmic-primary-30 flex md:hidden z-20"
      initial={{ y: 100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', damping: 20, stiffness: 300 }}
    >
      <nav className="w-full flex justify-around items-center">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `
              flex flex-col items-center justify-center px-1
              ${isActive 
                ? 'text-cosmic-highlight' 
                : 'text-gray-400 hover:text-cosmic-glow'
              }
              transition-colors
            `}
          >
            <div className={({ isActive }) => `
              p-1.5 rounded-full
              ${isActive ? 'bg-cosmic-primary-30' : ''}
            `}>
              {item.icon}
            </div>
            <span className="text-xs mt-0.5">{item.label}</span>
          </NavLink>
        ))}
      </nav>
    </motion.div>
  );
};

export default MobileNavbar;