import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { FaHome, FaUser, FaSignOutAlt, FaSignInAlt, FaBars, FaTimes, FaUserShield, FaUserCog, FaChartLine } from 'react-icons/fa';
import { useAuth } from '../utils/AuthContext';

/**
 * Barra de navegación principal con animaciones y estilo cósmico
 */
const NavBar = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  
  // Detectar scroll para cambiar estilo de la barra
  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 10;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [scrolled]);
  
  // Cerrar menú al cambiar de ruta
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);
  
  // Manejar cierre de sesión
  const handleLogout = async () => {
    await logout();
    navigate('/');
  };
  
  // Determinar ruta de dashboard según el rol del usuario
  const getDashboardPath = () => {
    if (!user) return '/login';
    
    switch (user.role) {
      case 'super_admin':
        return '/super-admin';
      case 'admin':
        return '/admin';
      case 'investor':
        return '/investor';
      default:
        return '/';
    }
  };
  
  // Variantes para animaciones
  const navVariants = {
    hidden: { opacity: 0, y: -50 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.5,
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.3
      }
    }
  };
  
  const mobileMenuVariants = {
    hidden: { 
      opacity: 0, 
      y: -20,
      clipPath: 'circle(0% at calc(100% - 30px) 30px)'
    },
    visible: { 
      opacity: 1, 
      y: 0,
      clipPath: 'circle(150% at calc(100% - 30px) 30px)',
      transition: {
        type: 'spring',
        damping: 20,
        stiffness: 300,
        staggerChildren: 0.1
      }
    },
    exit: {
      opacity: 0,
      clipPath: 'circle(0% at calc(100% - 30px) 30px)',
      transition: {
        duration: 0.3
      }
    }
  };
  
  return (
    <motion.nav 
      className={`navbar ${scrolled ? 'navbar-scrolled' : ''}`}
      variants={navVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <svg 
            className="genesis-logo-nav" 
            viewBox="0 0 100 100" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <linearGradient id="navLogoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#0cc6de" />
                <stop offset="100%" stopColor="#9270ff" />
              </linearGradient>
              <filter id="navGlow">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>
            <g filter="url(#navGlow)">
              <path 
                d="M50,10 A40,40 0 1,0 50,90 A40,40 0 1,0 50,10 Z M50,20 A30,30 0 1,1 50,80 A30,30 0 1,1 50,20 Z" 
                fill="none" 
                stroke="url(#navLogoGradient)" 
                strokeWidth="2"
              />
              <path 
                d="M50,10 L50,20 M50,80 L50,90 M10,50 L20,50 M80,50 L90,50" 
                stroke="url(#navLogoGradient)" 
                strokeWidth="2"
              />
              <path 
                d="M30,50 L70,50 M50,30 L50,70" 
                stroke="url(#navLogoGradient)" 
                strokeWidth="4"
              />
              <circle 
                cx="50" 
                cy="50" 
                r="8" 
                fill="url(#navLogoGradient)"
              />
            </g>
          </svg>
          <span className="logo-text">GÉNESIS</span>
        </Link>
        
        {/* Menú para escritorio */}
        <div className="navbar-links">
          <motion.div 
            className="nav-item"
            variants={itemVariants}
          >
            <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
              <FaHome />
              <span>Inicio</span>
            </Link>
          </motion.div>
          
          {isAuthenticated ? (
            <>
              <motion.div 
                className="nav-item"
                variants={itemVariants}
              >
                <Link 
                  to={getDashboardPath()} 
                  className={location.pathname.includes(getDashboardPath()) ? 'active' : ''}
                >
                  {user.role === 'super_admin' && <FaUserShield />}
                  {user.role === 'admin' && <FaUserCog />}
                  {user.role === 'investor' && <FaChartLine />}
                  <span>Panel</span>
                </Link>
              </motion.div>
              
              <motion.div 
                className="nav-item"
                variants={itemVariants}
              >
                <button onClick={handleLogout} className="nav-button">
                  <FaSignOutAlt />
                  <span>Salir</span>
                </button>
              </motion.div>
            </>
          ) : (
            <motion.div 
              className="nav-item"
              variants={itemVariants}
            >
              <Link 
                to="/login" 
                className={location.pathname === '/login' ? 'active' : ''}
              >
                <FaSignInAlt />
                <span>Ingresar</span>
              </Link>
            </motion.div>
          )}
        </div>
        
        {/* Botón menú móvil */}
        <div className="mobile-menu-button">
          <button onClick={() => setIsMenuOpen(!isMenuOpen)}>
            {isMenuOpen ? <FaTimes /> : <FaBars />}
          </button>
        </div>
        
        {/* Menú móvil */}
        <AnimatePresence>
          {isMenuOpen && (
            <motion.div 
              className="mobile-menu"
              variants={mobileMenuVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
              <motion.div variants={itemVariants} className="mobile-nav-item">
                <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
                  <FaHome />
                  <span>Inicio</span>
                </Link>
              </motion.div>
              
              {isAuthenticated ? (
                <>
                  <motion.div variants={itemVariants} className="mobile-nav-item">
                    <Link 
                      to={getDashboardPath()} 
                      className={location.pathname.includes(getDashboardPath()) ? 'active' : ''}
                    >
                      {user.role === 'super_admin' && <FaUserShield />}
                      {user.role === 'admin' && <FaUserCog />}
                      {user.role === 'investor' && <FaChartLine />}
                      <span>Panel</span>
                    </Link>
                  </motion.div>
                  
                  <motion.div variants={itemVariants} className="mobile-nav-item">
                    <button onClick={handleLogout} className="nav-button">
                      <FaSignOutAlt />
                      <span>Salir</span>
                    </button>
                  </motion.div>
                </>
              ) : (
                <motion.div variants={itemVariants} className="mobile-nav-item">
                  <Link 
                    to="/login" 
                    className={location.pathname === '/login' ? 'active' : ''}
                  >
                    <FaSignInAlt />
                    <span>Ingresar</span>
                  </Link>
                </motion.div>
              )}
              
              {isAuthenticated && (
                <motion.div variants={itemVariants} className="user-info">
                  <div className="user-avatar">
                    <FaUser />
                  </div>
                  <div className="user-details">
                    <span className="user-name">{user.username}</span>
                    <span className="user-role">
                      {user.role === 'super_admin' && 'Super Administrador'}
                      {user.role === 'admin' && 'Administrador'}
                      {user.role === 'investor' && 'Inversor'}
                    </span>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.nav>
  );
};

// Estilos específicos para el NavBar
const styles = `
  .navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 70px;
    background: rgba(13, 25, 48, 0.5);
    backdrop-filter: blur(10px);
    z-index: 100;
    transition: all var(--transition-normal);
  }
  
  .navbar-scrolled {
    background: rgba(13, 25, 48, 0.8);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }
  
  .navbar-container {
    max-width: 1200px;
    height: 100%;
    margin: 0 auto;
    padding: 0 var(--spacing-lg);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .navbar-logo {
    display: flex;
    align-items: center;
    text-decoration: none;
    gap: var(--spacing-xs);
  }
  
  .genesis-logo-nav {
    width: 40px;
    height: 40px;
  }
  
  .logo-text {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--color-primary);
    letter-spacing: 2px;
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  
  .navbar-links {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
  }
  
  .nav-item a, .nav-item button {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    color: var(--color-text);
    text-decoration: none;
    font-family: var(--font-display);
    font-weight: 500;
    border-radius: var(--border-radius-md);
    transition: all var(--transition-fast);
    border: none;
    background: none;
    cursor: pointer;
    font-size: 1rem;
  }
  
  .nav-item a:hover, .nav-item button:hover {
    color: var(--color-primary);
    background: rgba(12, 198, 222, 0.1);
  }
  
  .nav-item a.active {
    color: var(--color-primary);
    background: rgba(12, 198, 222, 0.1);
    position: relative;
  }
  
  .nav-item a.active::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 20px;
    height: 3px;
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
    border-radius: 3px;
  }
  
  .mobile-menu-button {
    display: none;
  }
  
  .mobile-menu-button button {
    background: none;
    border: none;
    font-size: 1.5rem;
    color: var(--color-text);
    cursor: pointer;
    padding: var(--spacing-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color var(--transition-fast);
  }
  
  .mobile-menu-button button:hover {
    color: var(--color-primary);
  }
  
  .mobile-menu {
    display: none;
    position: absolute;
    top: 70px;
    right: 0;
    width: 250px;
    background: rgba(13, 25, 48, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 0 0 0 var(--border-radius-lg);
    border-left: 1px solid var(--color-border);
    border-bottom: 1px solid var(--color-border);
    padding: var(--spacing-md);
    box-shadow: -5px 5px 20px rgba(0, 0, 0, 0.3);
  }
  
  .mobile-nav-item {
    margin-bottom: var(--spacing-sm);
  }
  
  .mobile-nav-item a, .mobile-nav-item button {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm) var(--spacing-md);
    color: var(--color-text);
    text-decoration: none;
    font-family: var(--font-display);
    font-weight: 500;
    border-radius: var(--border-radius-md);
    transition: all var(--transition-fast);
    width: 100%;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 1rem;
    text-align: left;
  }
  
  .mobile-nav-item a:hover, .mobile-nav-item button:hover {
    color: var(--color-primary);
    background: rgba(12, 198, 222, 0.1);
  }
  
  .mobile-nav-item a.active {
    color: var(--color-primary);
    background: rgba(12, 198, 222, 0.1);
  }
  
  .user-info {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    margin-top: var(--spacing-md);
    padding-top: var(--spacing-md);
    border-top: 1px solid var(--color-border);
  }
  
  .user-avatar {
    width: 40px;
    height: 40px;
    border-radius: var(--border-radius-circle);
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-background);
  }
  
  .user-details {
    display: flex;
    flex-direction: column;
  }
  
  .user-name {
    color: var(--color-text);
    font-weight: 600;
  }
  
  .user-role {
    color: var(--color-text-secondary);
    font-size: 0.8rem;
  }
  
  @media (max-width: 768px) {
    .navbar-links {
      display: none;
    }
    
    .mobile-menu-button {
      display: block;
    }
    
    .mobile-menu {
      display: block;
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'navbar-component-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}

export default NavBar;