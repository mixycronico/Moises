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
      className="fixed bottom-0 left-0 right-0 h-16 bg-cosmic-primary-20 backdrop-blur-xl border-t border-cosmic-primary-30 flex md:hidden z-20 shadow-lg"
      initial={{ y: 100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', damping: 20, stiffness: 300 }}
      style={{
        backgroundImage: 'linear-gradient(to top, rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.6))',
        boxShadow: '0 -4px 20px rgba(0, 0, 0, 0.25)'
      }}
    >
      <nav className="w-full max-w-screen-sm mx-auto flex justify-around items-center px-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `
              flex flex-col items-center justify-center px-2 py-1 
              ${isActive 
                ? 'text-cosmic-highlight' 
                : 'text-gray-400 hover:text-cosmic-glow'
              }
              transition-all duration-200 ease-in-out relative
            `}
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.div 
                    layoutId="nav-bubble"
                    className="absolute inset-0 bg-cosmic-primary-30 rounded-xl"
                    transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  />
                )}
                <div className="relative z-10 flex flex-col items-center justify-center">
                  <div className={`
                    p-1.5 rounded-full
                    ${isActive ? 'text-cosmic-highlight' : 'text-gray-400'}
                  `}>
                    {item.icon}
                  </div>
                  <span className="text-xs mt-0.5 font-medium">{item.label}</span>
                </div>
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </motion.div>
  );
};

export default MobileNavbar;